import math
import numpy as np
import random
import heapq


class OnlineDeliveryPlanner:

	def __init__(self, todo, max_distance, max_steering, verbose = False):

		self.todo = todo # Number of boxes to delivery
		self.max_distance = max_distance - 0.0001
		self.max_steering = max_steering - 0.0001
		self.verbose = verbose

		# Warehouse Setup
		self.warehouse = []
		self.warehouse_size = 201	
		self.dropzone_pos_world = (0. , 0.)
		self.dropzone_pos_grid = (self.warehouse_size/2, self.warehouse_size/2)
		# min_x, max_x, min_y, max_y
		self.warehouse_range = [self.dropzone_pos_grid[0], self.dropzone_pos_grid[0], self.dropzone_pos_grid[1], self.dropzone_pos_grid[1]] 
		self.wall_size = 0.5

		# Landmark Setup
		self.landmarks = dict()
		self.box_size = 0.1
		self.boxes_delivered = []
		self.target_range = 0.5

		# Robot setup
		self.robot_radius = 0.25
		self.robot_is_crashed = False
		self.robot_has_box = False

		self.robot_heading = 0.
		self.robot_pos_world = self.dropzone_pos_world
		self.robot_pos_grid = self.dropzone_pos_grid
		self.target = None
		self.path = []

		self.omega = np.zeros((2,2))
		self.omega[0][0] = 1.0e99
		self.omega[1][1] = 1.0e99
		self.Xi = np.zeros((2,1))
		self.mu = None
		self.last_measurement = None
		self.to_explore = set()

		self.last_action = (None, None, None)
		self.stuck_count = None


	def process_measurement(self, data, verbose = False):
		if verbose:
			# This is what the measurement data look like
			print "Obs \tID    Dist    Brg (deg)"
			for meas in data:
				# label the marker
				print "{} \t{}     {:6.2f}    {:6.2f}".format(meas[0],meas[1], meas[2],math.degrees(meas[3]))

		if not self.landmarks:
			self.landmarks['self'] = ['self', 0, None, None, 0]

		idx = len(self.landmarks)
		new_landmarks = dict()
		for meas in data:
			if (meas[1] not in self.landmarks):
				new_landmarks[meas[1]] = [meas[0], 2*idx, None, None, np.inf]
				idx += 1

		if new_landmarks:
			self.omega = np.pad(self.omega, ((0, 2*len(new_landmarks)), (0, 2*len(new_landmarks))), mode='constant', constant_values = 0)
			self.Xi = np.pad(self.Xi, ((0, 2*len(new_landmarks)), (0,0)), mode='constant', constant_values=0)
			self.landmarks.update(new_landmarks)

		for measurement in data:
			distance = measurement[2]
			angle = measurement[3]
			measurement_noise = math.exp(distance)

			dx = distance*math.cos(self.truncate_angle(angle+self.robot_heading))
			dy = distance*math.sin(self.truncate_angle(angle+self.robot_heading))
			meas_xy = (measurement[0], measurement[1], dx, dy)

			m = self.landmarks[meas_xy[1]][1]
			for b in range(2):
				self.omega[b][b] += 1.0/measurement_noise
				self.omega[m+b][m+b] += 1.0/measurement_noise
				self.omega[b][m+b] += -1.0/measurement_noise
				self.omega[m+b][b] += -1.0/measurement_noise

				self.Xi[b][0] += -meas_xy[2+b]/measurement_noise
				self.Xi[m+b][0] += meas_xy[2+b]/measurement_noise

		self.mu = np.matmul(np.linalg.inv(self.omega), self.Xi)

		self.robot_pos_world = self.get_landmark_pos('self')
		self.robot_pos_grid = self.get_landmark_pos('self', True)

		# Calculate the robot's heading
		x_vec = math.cos(self.robot_heading)
		y_vec = math.sin(self.robot_heading)
		for measurement in data:
			landmark_id = measurement[1]
			lm_x, lm_y = self.get_landmark_pos(landmark_id)
			dx = lm_x - self.robot_pos_world[0]
			dy = lm_y - self.robot_pos_world[1]
			lm_angle = math.atan2(dy,dx)
			est_robot_angle = self.truncate_angle(lm_angle - measurement[3])
			x_vec += math.cos(est_robot_angle)/measurement[2]
			y_vec += math.sin(est_robot_angle)/measurement[2]
			self.landmarks[landmark_id][4] = min(self.landmarks[landmark_id][4], math.sqrt(dx**2 + dy**2))

		self.robot_heading = math.atan2(y_vec, x_vec)
		# update the map
		self.update_map(data)
		self.last_measurement = data

	def next_move(self, verbose = False):
		# line-of-sight package
		los_package = [(self.compute_distance(self.get_landmark_pos(m[1]), self.robot_pos_world), m[1])
						for m in self.last_measurement if m[0] == 'box']

		if not los_package:
			if self.robot_pos_grid in self.to_explore:
				self.to_explore.remove(self.robot_pos_grid)
		elif self.robot_has_box:
			self.to_explore.add(self.robot_pos_grid)

		if self.path and self.robot_pos_grid == self.path[0] and \
			self.compute_distance(self.grid2coord(self.robot_pos_grid), self.robot_pos_world) < 0.25:
			square = self.path.pop(0)

		# Robot Status
		if self.robot_has_box:
			# Go back to start point
			self.target = '@'
			if not self.path:
				self.path = self.find_path(self.dropzone_pos_grid, True)
				# If stuck, random walk
				if self.path is None:
					self.path = [self.get_open_square()]
					self.target = None
		elif los_package:
			min_target = min(los_package)[1]
			self.path = self.find_path(self.get_landmark_pos(min_target, grid = True))
			if self.path is None:
				self.path = [self.get_open_square()]
				self.target = None
			else:
				self.target = min_target
		elif not self.path:
			if self.to_explore:
				square = next(iter(self.to_explore))
				self.path = self.find_path(square)
				if not self.path:
					self.path = self.explore_path()
			else:
				self.path = self.explore_path()

		if self.robot_is_crashed:
			self.route = [self.get_open_square()]

		target_pos_world = self.get_landmark_pos(self.target) if self.target else None
		target_pos_grid = self.get_landmark_pos(self.target, True) if target_pos_world else None


		if self.path:
			pos= self.path[0]
			if self.warehouse[pos[0]][pos[1]] == '#':
				if self.target:
					self.path = self.find_path(target_pos_grid, True if self.target is '@' else False)
				else:
					self.path = self.explore_path

			if self.path is None:
				self.path = [self.get_open_square()]
				self.target = None

			next_pos_world = self.grid2coord(self.path[0])

			if self.target and self.target is not '@' and len(self.path) == 1 and not self.robot_is_crashed:
				min_x = min(self.robot_pos_world[0], next_pos_world[0]) - self.robot_radius
				min_y = min(self.robot_pos_world[1], next_pos_world[1]) - self.robot_radius
				max_x = max(self.robot_pos_world[0], next_pos_world[0]) + self.robot_radius
				max_y = max(self.robot_pos_world[1], next_pos_world[1]) + self.robot_radius

				if min_x <= target_pos_world[0] <= max_x and min_y <= target_pos_world[1] <= max_y:
					self.route = []	

		if not self.path:
			if self.target == '@':
				next_pos_world = (0., 0.)
			else:
				next_pos_world = target_pos_world
				for neighbor in [n[1] for n in los_package if n[1] != self.target]:
					neighbor_pos_world = self.get_landmark_pos(neighbor)
					if self.compute_distance(next_pos_world, neighbor_pos_world) < 0.2:
						next_pos_world_x = (next_pos_world[0] + neighbor_pos_world[0])/2
						next_pos_world_y = (next_pos_world[1] + neighbor_pos_world[1])/2
						next_pos_world = [next_pos_world_x, next_pos_world_y]
						break

		bearing = self.truncate_angle(math.atan2(next_pos_world[1] - self.robot_pos_world[1], next_pos_world[0] - self.robot_pos_world[0]) - self.robot_heading)
		steering = min(bearing, self.max_steering)
		distance = self.compute_distance(next_pos_world, self.robot_pos_world)

		# send out mvoe command
		bearing = self.truncate_angle(math.atan2(next_pos_world[1] - self.robot_pos_world[1], next_pos_world[0] - self.robot_pos_world[0]) - self.robot_heading)
		distance = self.compute_distance(next_pos_world, self.robot_pos_world)
		if self.target and not self.target == '@' and not self.path:
			distance += self.box_size
		if self.robot_is_crashed:
			action = 'move'
			steering = min(self.max_steering, math.pi/4)
			distance = 0.1
		elif (self.target or self.robot_has_box) and distance < self.target_range and not self.path:
			if self.robot_has_box:
				action = 'down'
			else:
				action = 'lift'
			steering = 0.
			distance = 0.
		else:
			action = 'move'
			if math.fabs(bearing) > self.max_steering:
				angle_sign = np.sign(bearing)
				steering = self.truncate_angle(angle_sign*self.max_steering)
				distance = 0.0
			else:
				steering = bearing
				if self.target and not self.path:
					if self.target == '@':
						distance = min(distance - (self.target_range + 0.2)/2, self.max_distance)
					else:
						distance = min(distance - (self.target_range + self.robot_radius + self.box_size)/2, self.max_distance)
				else:
					distance = min(distance, self.max_distance)

		if action == 'move':
			move = '{action} {steering} {distance}'.format(action = action, steering = steering, distance = distance)
		else:
			move = '{action} {bearing}'.format(action = action, bearing = bearing)
		self.last_action = (action, steering, distance)

		self.online_slam_motion(steering, distance)
		return move

	def online_slam_motion(self, steering, dist):


		self.robot_heading = self.truncate_angle(self.robot_heading + steering)
		# move: (dx, dy)
		move = (dist*math.cos(self.robot_heading), dist*math.sin(self.robot_heading))
		motion_noise = 0.02

		dim = len(self.omega)
		self.omega = np.insert(self.omega, (2,2), values = 0, axis = 1)
		self.omega = np.insert(self.omega, 2, values=np.zeros((2, dim+2)), axis = 0)
		self.Xi = np.insert(self.Xi, 2, values=np.zeros((2,1)), axis = 0)

		for b in range(4):
			self.omega[b][b] += 1.0/motion_noise

		for b in range(2):
			self.omega[b][b+2] += -1.0/motion_noise
			self.omega[b+2][b] += -1.0/motion_noise
			self.Xi[b][0] += -move[b]/motion_noise
			self.Xi[b+2][0] += move[b]/motion_noise

		a = self.omega[:2, 2:]
		b = self.omega[:2, :2]
		c = self.Xi[:2, ]
		self.omega = self.omega[2:, 2:] - np.matmul(np.matmul(np.transpose(a), np.linalg.inv(b)), a)
		self.Xi = self.Xi[2:, ] - np.matmul(np.matmul(np.transpose(a), np.linalg.inv(b)), c)

	def set_robot_state(self, robot_has_box, robot_is_crashed, boxes_delivered, verbose = False):

		self.robot_has_box = robot_has_box
		self.robot_is_crashed = robot_is_crashed
		self.boxes_delivered = boxes_delivered

		if verbose:
			print "Robot has box: {},  Robot is crashed: {}".format(robot_has_box, robot_is_crashed)
			print "Boxes delivered: {}".format(boxes_delivered) 

		if self.robot_is_crashed:
			if self.stuck_count is None:
				self.stuck_count = 0
			self.stuck_count += 1
		else:
			self.stuck_count = None

		if self.stuck_count > 100:
			raise RuntimeError('Robot could not get unstucked after 100 iterations')

		if self.last_action[0] == 'down':
			if len(boxes_delivered) > len(self.boxes_delivered):
				self.target = None
			else:
				if verbose:
					print "Down failed"
		self.boxes_delivered = boxes_delivered


	def truncate_angle(self, t):
		while t < 0.0:
			t += 2*np.pi 

		return ((t+np.pi)%(2*np.pi)) - np.pi 

	def compute_distance(self, point1, point2):

		return math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

	def coord2grid(self, coordinate):
		row = int(-coordinate[1] + 0.5 + self.dropzone_pos_grid[0]) #row number
		col = int(coordinate[0] + 0.5 + self.dropzone_pos_grid[1]) #col number

		return row, col

	def grid2coord(self, grid):
		y = self.dropzone_pos_grid[0] - grid[0]
		x = grid[1] - self.dropzone_pos_grid[1]

		return x, y

	def get_landmark_pos(self, landmark_id, grid = False):
		if landmark_id is '@':
			pos = 0., 0.
		else:
			idx = self.landmarks[landmark_id][1]
			pos = [self.mu[idx][0], self.mu[idx+1][0]]
		
		if grid:
			return self.coord2grid(pos)
		else:
			return pos

	def align(self, pos):
		x, y = pos
		ax = round(x*2)/2
		ay = round(y*2)/2
		return [ax, ay]


	def update_map(self, measurement):

		for m in measurement:

			landmark_type = m[0]
			landmark_id = m[1]

			if landmark_type in {'wall', 'warehouse'}:
				pass
			elif landmark_type in {'box'}:
				continue

			landmark_pos_world = self.get_landmark_pos(landmark_id)
			landmark_pos_world = self.align(landmark_pos_world)

			if landmark_pos_world[0] > self.robot_pos_world[0]:
				if landmark_pos_world[1] == self.robot_pos_world[1] or not landmark_pos_world[0].is_integer():
					face = 'w'
				elif landmark_pos_world[1] > self.robot_pos_world[1]:
					face = 's'
				else:
					face = 'n'
			elif landmark_pos_world[0] == self.robot_pos_world[0]:
				if landmark_pos_world[1] > self.robot_pos_world[1]:
					face = 's'
				else:
					face = 'n'
			else:
				if landmark_pos_world[1] == self.robot_pos_world[1] or not landmark_pos_world[0].is_integer():
					face = 'e'
				elif landmark_pos_world[1] > self.robot_pos_world[1]:
					face = 's'
				else:
					face = 'n'

			self.landmarks[landmark_id][2] = face


		self.warehouse = [['.' for col in range(self.warehouse_size)] for row in range(self.warehouse_size)]
		self.warehouse_range = [self.dropzone_pos_grid[0], self.dropzone_pos_grid[0], self.dropzone_pos_grid[1], self.dropzone_pos_grid[1]] 

		for key, value in self.landmarks.items():

			if value[0] not in {'wall', 'warehouse'}:
				continue
			#print key, value

			landmark_pos = self.get_landmark_pos(key)
			landmark_pos = self.align(landmark_pos)

			face = value[2]
			if face == 'n':
				landmark_pos[1] -= self.wall_size
			elif face == 's':
				landmark_pos[1] += self.wall_size
			elif face == 'w':
				landmark_pos[0] += self.wall_size
			elif face == 'e':
				landmark_pos[0] -= self.wall_size

			landmark_pos_grid = self.coord2grid(landmark_pos)
			#print landmark_pos_grid

			if landmark_pos_grid[0] > self.warehouse_range[1]:
				self.warehouse_range[1] = landmark_pos_grid[0]
			if landmark_pos_grid[0] < self.warehouse_range[0]:
				self.warehouse_range[0] = landmark_pos_grid[0]
			if landmark_pos_grid[1] > self.warehouse_range[3]:
				self.warehouse_range[3] = landmark_pos_grid[1]
			if landmark_pos_grid[1] < self.warehouse_range[2]:
				self.warehouse_range[2] = landmark_pos_grid[1]

			self.warehouse[landmark_pos_grid[0]][landmark_pos_grid[1]] = '#'
			self.landmarks[key][3] = landmark_pos_grid

	def find_path(self, target_pos, find_orthogonal_pos = False):

		f = self.compute_distance(target_pos, self.robot_pos_grid)
		i,j = self.robot_pos_grid
		g = 0
		open_nodes = []
		heapq.heappush(open_nodes, [f,g,i,j])
		closed = set()
		closed.add((i,j))
		action = [[-1 for col in range(len(self.warehouse[0]))] for row in range(len(self.warehouse))]
		action[i][j] = 0,0
		found = False
		move_delta = [(-1,0), (1,0),(0,-1),(0,1)]

		while not found:
			if not open_nodes:
				return None

			next_node = heapq.heappop(open_nodes)
			i = next_node[2]
			j = next_node[3]
			g = next_node[1]

			if (i, j) == target_pos:
				found = True
				break

			for (di, dj) in move_delta:
				i2 = i + di
				j2 = j + dj
				if i2 >= len(self.warehouse) or j2 >= len(self.warehouse[0]) or i2 < 0 or j2 < 0:
					continue
				if (i2, j2) in closed:
					continue
				if self.warehouse[i2][j2] == '#' and not (find_orthogonal_pos and (i2, j2) == target_pos):
					continue
				if find_orthogonal_pos and (i2, j2) == target_pos:
					found = True
					break

				g2 = g + self.compute_distance((i2, j2), (i, j))
				h2 = self.compute_distance((i2, j2), target_pos)
				f2 = g2 + h2
				heapq.heappush(open_nodes, (f2, g2, i2, j2))
				closed.add((i2, j2))
				action[i2][j2] = (di, dj)
		

		locations = []
		while (i,j) != self.robot_pos_grid:
			locations.insert(0, (i,j))
			i2 = i - action[i][j][0]
			j2 = j - action[i][j][1]
			i, j = i2, j2

		if locations and self.target == '@':
			next_pos_world = self.grid2coord(locations[0])
			est_next_pos = self.grid2coord(self.robot_pos_grid)
			if self.compute_distance(self.robot_pos_world, est_next_pos) > 0.25 and \
				self.compute_distance(est_next_pos, next_pos_world) < self.compute_distance(self.robot_pos_world, next_pos_world):
				locations.insert(0,(i,j))
		return locations

	def explore_path(self):
		landmark_dist = sorted((v[4], v[3],v[2], k) for k, v in self.landmarks.items() if k is not 'self')
		while landmark_dist:
			landmark = landmark_dist.pop()
			target_i, target_j = landmark[1]

			landmark_face = landmark[2]
			if landmark_face == 'n':
				target_i -= 1
			elif landmark_face == 's':
				target_i += 1
			elif landmark_face == 'e':
				target_j += 1
			elif landmark_face == 'w':
				target_j -= 1

			path = self.find_path((target_i, target_j))
			if path:
				return path

		path = None
		while not path:
			i = random.randint(self.warehouse_range[0]+1, self.warehouse_range[1]-1)
			j = random.randint(self.warehouse_range[2]+1, self.warehouse_range[3]-1)
			if self.warehouse[i][j] == '#':
				continue
			path = self.find_path((i,j))

		return path
		
	def get_open_square(self):
		move_delta =[(1,0),(-1,0),(0,1),(0,-1)]
		select = random.sample(move_delta, len(move_delta))
		next_square = None
		for delta in select:
			next_square = (self.robot_pos_grid[0] + delta[0], self.robot_pos_grid[1] + delta[1])
			if self.warehouse[next_square[0]][next_square[1]] != '#':
				return next_square
		return next_square









