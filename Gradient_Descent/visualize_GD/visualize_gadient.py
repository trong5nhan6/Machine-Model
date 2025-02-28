import pygame
from random import randint
from sklearn import linear_model
import numpy as np

class TextButton:
	def __init__(self, text, position, circle_pos, radius):
		self.text = text
		self.position = position
		self.circle_pos = circle_pos
		self.radius = radius

	def is_mouse_on_text(self):
		mouse_x, mouse_y = pygame.mouse.get_pos()
		if self.circle_pos[0] - self.radius < mouse_x < self.circle_pos[0] + self.radius and self.circle_pos[1] - self.radius < mouse_y < self.circle_pos[1] + self.radius :
			return True
		else:
			return False

	def draw(self):
		font = pygame.font.SysFont('sans', 40)
		text_render = font.render(self.text, True, (255,255,255))
		self.text_box = text_render.get_rect()
		pygame.draw.circle(screen,WHITE, self.circle_pos, self.radius, 1)

		if self.is_mouse_on_text():
			text_render = font.render(self.text, True, (0,255,255))
			pygame.draw.line(screen, (0,255,255), (self.position[0], self.position[1] + self.text_box[3]), (self.position[0] + self.text_box[2], self.position[1] + self.text_box[3]))
			pygame.draw.circle(screen,BLUE, self.circle_pos, self.radius, 1)
		else:
			text_render = font.render(self.text, True, (255,255,255))

		screen.blit(text_render, self.position)


# gradient fomar
def cost(x):
	m = A.shape[0]
	return 0.5/m * np.linalg.norm(A.dot(x) - b, 2)**2

def grad(x):
	m = A.shape[0]
	return 1/m * A.T.dot(A.dot(x)-b)

def check_grad(x):
	eps = 1e-4
	g = np.zeros_like(x)	
	for i in range(len(x)):
		x1 = x.copy()
		x2 = x.copy()
		x1[i] += eps
		x2[i] -= eps
		g[i] = (cost(x1) - cost(x2))/(2*eps)	

	g_grad = grad(x)
	if np.linalg.norm(g-g_grad) > 1e-5:
		print("WARNING: CHECK GRADIENT FUNCTION!")

def gradient_descent(x_init, learning_rate, iteration):
	x_list = [x_init]
	m = A.shape[0]
	# ones = np.ones((A.shape[0],1), dtype=np.int8)
	# A = np.concatenate((ones,A), axis=1)

	for i in range(iteration):
		x_new = x_list[-1] - learning_rate*grad(x_list[-1])
		if np.linalg.norm(grad(x_new))/m < 0.5: # when to stop GD
			break
		x_list.append(x_new)

	return x_list


# create A,b
def create_vector(number):
	A = []
	for i in range(len(points)):
		A.append([points[i][number]])
	A = np.array(A)
	return A


def linear_regression(A,b):
	A = np.array(A)
	b = np.array(b)
	# Create model
	lr = linear_model.LinearRegression()
	# Fit (train the model)
	lr.fit(A,b)
	x0 = np.array([[1,1000]]).T
	y0 = x0*lr.coef_ + lr.intercept_
	point_1 = [1]
	point_1.append(y0[0][0])
	point_2 = [1000]
	point_2.append(y0[1][0])
	return point_1, point_2


pygame.init()
screen = pygame.display.set_mode((1000, 700))
pygame.display.set_caption('Flappy Bird')
running = True
clock = pygame.time.Clock()

GREEN = (0,255,0)
BLACK = (0,0,0)
WHITE = (255,255,255)
BLUE = (0,255,255)
PINK = (255,20,147)

points = []	
line_points = []
A = []
b = []
Alg = []

reset_text_pos = (60,575)
back_text_pos = (265,575)
random_line_text_pos = (435,575)
GD_text_pos = (675,575)
run_text_pos = (870,575)

reset_circle_pos = (100, 600)
back_circle_pos = (300, 600)
random_line_circle_pos = (500,600)
GD_circle_pos = (700,600)
run_circle_pos = (900, 600)

radius = 75

reset_btn = TextButton("Reset",reset_text_pos,reset_circle_pos,radius)
back_btn = TextButton("Back",back_text_pos,back_circle_pos,radius)
random_line_btn = TextButton("Ran_line",random_line_text_pos,random_line_circle_pos,radius)
GD_btn = TextButton("GD",GD_text_pos,GD_circle_pos,radius)
run_btn = TextButton("Run",run_text_pos,run_circle_pos,radius)

pausing = False
GD_pasing = False

# Run gradient descent
iteration = 100
learning_rate = 0.0001

y0_xlist_list = []
# x_init = []
# x0 = np.array([[line_points[0][0],line_points[1][0]]]).T
# x0 = np.array([[1,1000]]).T

number_line_GD = 0

while running:		
	clock.tick(60)
	screen.fill(BLACK)

	mouse_x, mouse_y = pygame.mouse.get_pos()

	pygame.draw.line(screen,WHITE,(0,500),(1000,500))

	# draw reset button
	reset_btn.draw()

	# draw back button
	back_btn.draw()

	# draw line button
	random_line_btn.draw()

	# draw GD button
	GD_btn.draw()

	# draw run button
	run_btn.draw()

	# create A,b
	A = create_vector(0)
	b = create_vector(1)

	if len(A) != 0:
		# Add one to A
		ones = np.ones((A.shape[0],1), dtype=np.int8)
		A = np.concatenate((A,ones), axis=1)

	# check line_points exit
	if len(line_points) == 2:
		GD_pasing = True
		x0 = np.array([[line_points[0][0],line_points[1][0]]]).T
	else: 
		GD_pasing = False


	if GD_pasing == True and len(points) != 0:

		# Cho trước các giá trị
		x0_gd = np.linspace(line_points[0][0], line_points[1][0], 2)
		y0_init = np.array([line_points[0][1], line_points[1][0]])

		# Tạo ma trận hệ số D từ x0_gd
		D = np.vstack([x0_gd, np.ones_like(x0_gd)]).T

		x_init = np.linalg.solve(D, y0_init)
		x_init = np.array([x_init]).T

		# use gradient
		x_list = gradient_descent(x_init, learning_rate, iteration)

		# create y0_xlist_list (solution by GD)
		for i in range(len(x_list)):
			y0_xlist = x_list[i][0] + x_list[i][1]*x0_gd
			y0_xlist_list.append(y0_xlist)


	# draw line linear regression
	if Alg != None and pausing == True:
		pygame.draw.line(screen, WHITE,Alg[0],Alg[1],3)

	for event in pygame.event.get():
		if event.type == pygame.QUIT:
			running = False
				
		if event.type == pygame.MOUSEBUTTONDOWN:
			if event.button != 3:

				# Create point on panel
				if 0 < mouse_x < 1000 and 0 < mouse_y < 500:
					point = [mouse_x, mouse_y]
					points.append(point)

				if reset_btn.is_mouse_on_text():
					points = []
					line_points = []
					A = []
					b = []
					Alg = None

					print("reset button")

				if back_btn.is_mouse_on_text():
					# remove last list in np.array
					if len(points) != 0:
						del points[-1]
						A = np.delete(A, -1, axis=0)
						b = np.delete(b, -1, axis=0)

					if len(points) > 0:
						Alg = linear_regression(A,b)
					else : Alg = None

					print("back button")
				
				if random_line_btn.is_mouse_on_text():
					line_points = []

					print("random_line button")

				if GD_btn.is_mouse_on_text():

					if GD_pasing == True:
							# Cho trước các giá trị
						x0_gd = np.linspace(line_points[0][0], line_points[1][0], 2)
						y0_init = np.array([line_points[0][1], line_points[1][0]])

						# Tạo ma trận hệ số D từ x0_gd
						D = np.vstack([x0_gd, np.ones_like(x0_gd)]).T

						x_init = np.linalg.solve(D, y0_init)
						x_init = np.array([x_init]).T

						# use gradient
						x_list = gradient_descent(x_init, learning_rate, iteration)

						# create y0_xlist_list (solution by GD)
						for i in range(len(x_list)):
							y0_xlist = x_list[i][0] + x_list[i][1]*x0_gd
							y0_xlist_list.append(y0_xlist)

						pygame.draw.line(screen,PINK,(x0[0][0], y0_xlist_list[number_line_GD][0]),(x0[1][0], y0_xlist_list[number_line_GD][1]))
						number_line_GD += 1
						print(x0)
						print(len(y0_xlist_list))

					print("GD button")

				if run_btn.is_mouse_on_text():
					pausing = True
					if len(points) != 0:
						Alg = linear_regression(A,b)
					print("run button")

			if event.button == 3:
				if len(points) > 2:
					if len(line_points) < 2:
						line_points.append([mouse_x,mouse_y])

		# Draw point
	for i in range(len(points)):	
		pygame.draw.circle(screen,GREEN, (points[i][0], points[i][1]), 5)

		# draw point line
	for i in range(len(line_points)):	
		pygame.draw.circle(screen,PINK, (line_points[i][0], line_points[i][1]), 3)

		# draw random line
	if len(line_points) == 2:
		pygame.draw.line(screen,PINK,line_points[0],line_points[1],3)
		
	pygame.display.flip()

pygame.quit()
