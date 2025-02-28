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

def linear_regression(A,b):
	A = np.array(A)
	b = np.array(b)
	# Create model
	lr = linear_model.LinearRegression()
	# Fit (train the model)
	lr.fit(A,b)
	x0 = np.array([[1,800]]).T
	y0 = x0*lr.coef_ + lr.intercept_
	point_1 = [1]
	point_1.append(y0[0][0])
	point_2 = [800]
	point_2.append(y0[1][0])
	return point_1, point_2

def create_vector(number):
	A = []
	for i in range(len(points)):
		A.append([points[i][number]])
	return A

pygame.init()
screen = pygame.display.set_mode((1000, 600))
pygame.display.set_caption('Flappy Bird')
running = True
clock = pygame.time.Clock()
BLACK = (0, 0, 0)
GREEN = (0,255,0)
WHITE = (255,255,255)
BLUE = (0,255,255)

points = []	
A = [] 
b = [] 
Alg = None

font = pygame.font.SysFont('sans', 40)
text_run = font.render('Run', True, WHITE)
text_back = font.render('Back', True, WHITE)
text_reset = font.render('Reset', True, WHITE)
text_reset_box = text_reset.get_rect()
text_back_box = text_back.get_rect()
text_run_box = text_run.get_rect()

reset_pos = (855,70)
back_pos = (865,270)
run_pos = (870,470)

reset_circle = (900, 100)
back_circle = (900, 300)
run_circle = (900, 500)

radius = 75

reset_btn = TextButton("Reset",reset_pos,reset_circle,radius)
back_btn = TextButton("Back",back_pos,back_circle,radius)
run_btn = TextButton("Run",run_pos,run_circle,radius)

pausing = False
while running:		
	clock.tick(60)
	screen.fill(BLACK)

	# check position mouse
	mouse_x, mouse_y = pygame.mouse.get_pos()

	# draw line 
	pygame.draw.line(screen, WHITE, (800,0), (800,600))

	# create A and b
	A = create_vector(0)
	b = create_vector(1)

	# draw reset button
	reset_btn.draw()

	# draw back button
	back_btn.draw()

	# draw run button
	run_btn.draw()

	# check point to draw line
	if len(points) == 0:
		pausing = False

	# draw line linear regression
	if Alg != None and pausing == True:
		pygame.draw.line(screen, WHITE,Alg[0],Alg[1],3)

	for event in pygame.event.get():
		if event.type == pygame.QUIT:
			running = False

		if event.type == pygame.MOUSEBUTTONDOWN:
			# Create point on panel
			if 0 < mouse_x < 800 and 0 < mouse_y < 600:
				point = [mouse_x, mouse_y]
				points.append(point)
				# A.append([mouse_x])
				# b.append([mouse_y])
			
			if reset_btn.is_mouse_on_text():
				points = []
				A = []
				b = []
				Alg = None

				print("button reset pressed")

			if back_btn.is_mouse_on_text():

				# remove last list in np.array
				if len(points) != 0:
					del points[-1]
					# A = np.delete(A, np.where(np.all(A == np.array(A[-1]), axis=1)), axis=0)
					# b = np.delete(b, np.where(np.all(b == np.array(b[-1]), axis=1)), axis=0)

					A = np.delete(A, -1, axis=0)
					b = np.delete(b, -1, axis=0)

				if len(points) > 0:
					Alg = linear_regression(A,b)
				else : Alg = None

				print("button back pressed")

			# run button
			if run_btn.is_mouse_on_text():
				pausing = True
				if len(points) != 0:
					Alg = linear_regression(A,b)

					print("button run pressed")
								
	# Draw point
	for i in range(len(points)):	
		pygame.draw.circle(screen,GREEN, (points[i][0], points[i][1]), 5)

	pygame.display.flip()

pygame.quit()