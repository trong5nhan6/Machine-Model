import pygame
from random import randint
from sklearn import linear_model
import numpy as np
import matplotlib
import matplotlib.pyplot as plt


class TextButton:
	def __init__(self, text, rect_pos):
		self.text = text
		self.rect_pos = rect_pos
		self.btn_check = False
		self.text_input_active = False
		self.check_text = False
		self.input_text = ""
		self.number = 0
		
	def is_mouse_on_text(self):
		mouse_x, mouse_y = pygame.mouse.get_pos()
		if self.rect_pos[0] < mouse_x < self.rect_pos[0] + self.rect_pos[2] and self.rect_pos[1] < mouse_y < self.rect_pos[1] + self.rect_pos[3] :
			return True
		else:
			return False

	def position_text(self):
		dis_x = 0
		dis_y = 0
		WIDTH = (self.rect_pos[0], self.rect_pos[2])
		HEIGHT = (self.rect_pos[1], self.rect_pos[3])
		dis_x = (self.rect_pos[2] - self.text_box[2])/2
		dis_y = (self.rect_pos[3] - self.text_box[3])/2
		pos_text = (self.rect_pos[0] + dis_x, self.rect_pos[1] + dis_y)
		return pos_text

	def draw(self):
		font = pygame.font.SysFont('sans', 40)
		text_render = font.render(self.text, True, WHITE)
		self.text_box = text_render.get_rect()
		pygame.draw.rect(screen, BLACK, (self.rect_pos[0], self.rect_pos[1], self.rect_pos[2], self.rect_pos[3]))
		pygame.draw.rect(screen, WHITE, (self.rect_pos[0], self.rect_pos[1], self.rect_pos[2], self.rect_pos[3]),1)
		pos_text = self.position_text()

		if self.is_mouse_on_text():
			text_render = font.render(self.text, True, BLUE)
			pygame.draw.rect(screen, BLACK, (self.rect_pos[0], self.rect_pos[1], self.rect_pos[2], self.rect_pos[3]))
			pygame.draw.rect(screen, BLUE, (self.rect_pos[0], self.rect_pos[1], self.rect_pos[2], self.rect_pos[3]),2)
		else:
			text_render = font.render(self.text, True, WHITE)

		screen.blit(text_render, pos_text)

	def text_input(self, event):
		if self.is_mouse_on_text() and event.type == pygame.MOUSEBUTTONDOWN:
			self.text_input_active = True
			if self.text_input_active:
				self.text = self.input_text
		elif self.text_input_active and event.type == pygame.KEYDOWN:
			if event.key == pygame.K_RETURN:
				self.text_input_active = False
				self.text = self.input_text
			elif event.key == pygame.K_BACKSPACE:
				self.input_text = self.input_text[:-1]
			else:
				self.input_text += event.unicode


	# Thêm phương thức để kiểm tra và xử lý sự kiện chuột cho nút "iters"
	def handle_event(self, event):
		if event.type == pygame.MOUSEBUTTONDOWN and self.is_mouse_on_text():
			self.check_text = True
		elif event.type == pygame.KEYDOWN and self.check_text:
			if event.key == pygame.K_RETURN:
				self.check_text = False
			elif event.key == pygame.K_BACKSPACE:
				self.text = self.text[:-1]
			else:
				self.text += event.unicode

	def pick_text_input(self):
		if self.text.isdigit():
			self.number = self.text
		return self.number


# create A,b
def create_vector(ordinal, number):
	A = [[]]
	for i in range(len(points)):
		A[0].append(points[i][ordinal] - number)
	A = np.array(A).T
	return A

def create_vector_ones(A):
	B = A
	ones = np.ones((B.shape[0],1), dtype=np.int8)
	B = np.concatenate((ones,B), axis=1)
	return B

# linear regression fomular
def linear_regression(A,b):

	# Create model
	lr = linear_model.LinearRegression()

	# Fit (train the model)
	lr.fit(A,b)
	x0 = np.array([[50,950]]).T
	y0 = x0*lr.coef_ + lr.intercept_

	point_1 = [50]
	point_1.append(y0[0][0])
	point_2 = [950]
	point_2.append(y0[1][0])
	return point_1, point_2

# gradient fomular
def cost(x, matrix, vector):
	m = matrix.shape[0]
	return 0.5/m * np.linalg.norm(matrix.dot(x) - vector, 2)**2

def grad(x, matrix, vector):
	m = matrix.shape[0]
	return 1/m * matrix.T.dot(matrix.dot(x)-vector)

def check_grad(x, matrix, vector):
	eps = 1e-4
	g = np.zeros_like(x)	
	for i in range(len(x)):
		x1 = x.copy()
		x2 = x.copy()
		x1[i] += eps
		x2[i] -= eps
		g[i] = (cost(x1, matrix, vector) - cost(x2, matrix, vector))/(2*eps)	

	g_grad = grad(x)
	if np.linalg.norm(g-g_grad) > 1e-5:
		print("WARNING: CHECK GRADIENT FUNCTION!")

def gradient_descent(x_init, matrix, vector, learning_rate, iteration):
	x_list = [x_init]
	m = matrix.shape[0]

	for i in range(iteration):
		x_new = x_list[-1] - learning_rate*grad(x_list[-1], matrix, vector)
		# if np.linalg.norm(grad(x_new, matrix, vector), matrix, vector)/m < 0.5: # when to stop GD
		# 	break
		x_list.append(x_new)

	return x_list


pygame.init()
screen = pygame.display.set_mode((1200, 700))
pygame.display.set_caption('Flappy Bird')
running = True
GREEN = (0, 200, 0)
clock = pygame.time.Clock()

BACKGROUND = (214, 214, 214)
BLACK = (0,0,0)
BACKGROUND_PANEL = (249, 255, 230)
WHITE = (255,255,255)
BLUE = (0,255,255)
RED = (250,10,10)
GREEN = (10,250,10)

reset_rect_pos = (50,580,250,90)
LR_rect_pos = (375,580,250,90)
back_rect_pos = (700,580,250,90)

GD_rect_pos = (1000,50,150,80)
iters_rect_pos = (1000,185,150,80)
learning_rect_pos = (1000,320,150,80)
plus_rect_pos = (1000,455,150,80)
plot_rect_pos = (1000,590,150,80)

reset_btn = TextButton("Reset", reset_rect_pos)
LR_btn = TextButton("LR", LR_rect_pos)
back_btn = TextButton("Back", back_rect_pos)

GD_btn = TextButton("GD", GD_rect_pos)
iters_btn = TextButton("Iteration", iters_rect_pos)
learning_btn = TextButton("L_Rate", learning_rect_pos)
plus_btn = TextButton("+", plus_rect_pos)
plot_btn = TextButton("Plot", plot_rect_pos)

font_small = pygame.font.SysFont('sans', 20)

points = []	
A = []
b = []
LR_alg_py = []
LR_alg_plot = []
line_points = []

pausing = False
plot_pausing = False

# Run gradient descent
iteration = 100
learning_rate = 0.0001

y0_xlist_list = []

while running:		
	clock.tick(60)
	screen.fill(BACKGROUND)

	# Check mouse in display
	mouse_x, mouse_y = pygame.mouse.get_pos()

	# Draw interface
	# Draw panel
	pygame.draw.rect(screen, BLACK, (50,50,900,500))
	pygame.draw.rect(screen, BACKGROUND_PANEL, (55,55,890,490))

	# Draw under button
	reset_btn.draw()
	LR_btn.draw()
	back_btn.draw()

	# Draw right button 
	GD_btn.draw()
	iters_btn.draw()
	learning_btn.draw()
	plus_btn.draw()
	plot_btn.draw()

	# Draw mouse position when mouse is in panel
	if 50 < mouse_x < 950 and 50 < mouse_y < 550:
		text_mouse = font_small.render("(" + str(mouse_x) + "," + str(mouse_y) + ")",True, BLACK)
		screen.blit(text_mouse, (mouse_x + 10, mouse_y))

	# Create A,b for pygame
	A = create_vector(0,0)
	b = create_vector(1,0)

	# Create A,b for plot
	C = create_vector(0,50)
	d = create_vector(1,50)

	# Create A_ones with concatenate ones
	A_ones = create_vector_ones(A)
	C_ones = create_vector_ones(C)

	# draw line linear regression
	if len(LR_alg_py) != 0 and pausing == True:
		pygame.draw.line(screen, GREEN,LR_alg_py[0],LR_alg_py[1],3)


	for event in pygame.event.get():
		if event.type == pygame.QUIT:
			running = False

		# write input on iters_btn andlearning_btn
		iters_btn.handle_event(event)
		iters_btn.text_input(event)
		learning_btn.handle_event(event)
		learning_btn.text_input(event)

		if event.type == pygame.MOUSEBUTTONDOWN:

			# draw random line
			if event.button == 3 and 50 < mouse_x < 950 and 50 < mouse_y < 550:
				if len(points) >= 2:
					if len(line_points) < 2:
						line_points.append([mouse_x,mouse_y])

			# remove last list in line_points when click right mouse on back button
			if back_btn.is_mouse_on_text() and event.button == 3:
					if len(line_points) != 0:
						del line_points[-1]

			if event.button != 3:			
				# Create point on panel
				if 50 < mouse_x < 950 and 50 < mouse_y < 550:
					point = [mouse_x, mouse_y]
					points.append(point)

				if reset_btn.is_mouse_on_text():
					points = []	
					A = []
					b = []
					LR_alg_py = []
					LR_alg_plot = []
					line_points = []
					print("reset")

				if LR_btn.is_mouse_on_text():
					pausing = True
					if len(points) != 0:
						LR_alg_py = linear_regression(A,b)

					print("LR")

				if back_btn.is_mouse_on_text():
					# remove last list in np.array
					if len(points) != 0:
						del points[-1]
						A = np.delete(A, -1, axis=0)
						b = np.delete(b, -1, axis=0)

					if len(points) > 0:
						LR_alg_py = linear_regression(A,b)
					else :
						LR_alg_py = []

					print("back")

				if GD_btn.is_mouse_on_text():

					if line_points == 2:

						# Cho trước các giá trị
						x0_gd = np.linspace(line_points[0][0], line_points[1][0], 2)
						y0_init = np.array([line_points[0][1], line_points[1][0]])

						# Tạo ma trận hệ số D từ x0_gd
						D = np.vstack([np.ones_like(x0_gd), x0_gd]).T

						x_init = np.linalg.solve(D, y0_init)
						x_init = np.array([x_init]).T

						# use gradient
						x_list = gradient_descent(x_init, learning_rate, iteration)

						print(x_list)

						# create y0_xlist_list (solution by GD)
						for i in range(len(x_list)):
							y0_xlist = x_list[i][0] + x_list[i][1]*x0_gd
							y0_xlist_list.append(y0_xlist)


					print("GD")

				if iters_btn.is_mouse_on_text():

					print("iters")

				if learning_btn.is_mouse_on_text():
					print("learning")

				if plus_btn.is_mouse_on_text():
					print("plus")

				if plot_btn.is_mouse_on_text():

					if len(points) > 0:
						plot_pausing = True
					if len(points) == 0:
						plot_pausing = False

					if plot_pausing:

						plt.plot(C,d,'ro')
						LR_alg_plot = linear_regression(C,d)
						x_LR_points = np.array([LR_alg_plot[0][0], LR_alg_plot[1][0]])
						y_LR_points = np.array([LR_alg_plot[0][1], LR_alg_plot[1][1]])
						plt.plot(x_LR_points,y_LR_points)
						plt.show()

					if len(line_points) == 2:

						plt.plot(C,d,'ro')
						LR_alg_plot = linear_regression(C,d)
						x_LR_points = np.array([LR_alg_plot[0][0], LR_alg_plot[1][0]])
						y_LR_points = np.array([LR_alg_plot[0][1], LR_alg_plot[1][1]])
						plt.plot(x_LR_points,y_LR_points)

						# Cho trước các giá trị
						x0_gd = np.linspace(line_points[0][0], line_points[1][0], 2)
						y0_init = np.array([line_points[0][1], line_points[1][0]])

						# Tạo ma trận hệ số D từ x0_gd
						D = np.vstack([np.ones_like(x0_gd), x0_gd]).T

						x_init = np.linalg.solve(D, y0_init)
						x_init = np.array([x_init]).T

						# use gradient
						x_list = gradient_descent(x_init, C_ones, d, learning_rate, iteration)

						print(C)
						print("xxx")
						print(d)
						print("xxx")
						print(x_init)
						print("xxx")
						print(x0_gd)
						print("xxx")
						print(y0_init)
						print("xxx")
						print(x_list)

						# # create y0_xlist_list (solution by GD)
						# for i in range(len(x_list)):
						# 	y0_xlist = x_list[i][0] + x_list[i][1]*x0_gd
						# 	y0_xlist_list.append(y0_xlist)

						# plot black x_list
						for i in range(len(x_list)):
							y0_x_list = x_list[i][0] + x_list[i][1]*x0_gd
							plt.plot(x0_gd, y0_x_list, color='black', alpha = 0.3)

						plt.show()

					print("plot")

	# Draw point
	for i in range(len(points)):	
		pygame.draw.circle(screen, RED, (points[i][0], points[i][1]), 5)

	# draw point line
	for i in range(len(line_points)):	
		pygame.draw.circle(screen,BLACK, (line_points[i][0], line_points[i][1]), 3)

	# draw random line
	if len(line_points) == 2:
		pygame.draw.line(screen,BLACK,line_points[0],line_points[1],3)
				
	pygame.display.flip()

pygame.quit()