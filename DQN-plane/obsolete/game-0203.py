import gym
from gym import spaces
import pygame
import random
import pygame.mixer
import numpy as np
import itertools

class Plane(object):
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.image = pygame.image.load("material/plane.png")
        self.bullets = []

    def draw(self, screen):
        screen.blit(self.image, (self.x, self.y))
        for bullet in self.bullets:
            bullet.draw(screen)

class Bullet(object):
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.image = pygame.image.load("material/bullet.png")

    def draw(self, screen):
        screen.blit(self.image, (self.x, self.y))
        self.y -= 5

    def check_collision(self, enemy):
        if pygame.Rect(self.x, self.y, self.image.get_width(), self.image.get_height()).colliderect(pygame.Rect(enemy.x, enemy.y, enemy.image.get_width(), enemy.image.get_height())):
            return True
        else:
            return False

class Enemy(object):
    def __init__(self,x,y):
        self.x = x
        self.y = y
        self.image = pygame.image.load("material/enemy.png")

    def update(self):
        self.y += 2

    def draw(self, screen):
        screen.blit(self.image, (self.x, self.y))

    def check_collision(self, plane):
        if pygame.Rect(plane.x, plane.y, plane.image.get_width(), plane.image.get_height()).colliderect(pygame.Rect(self.x, self.y, self.image.get_width(), self.image.get_height())):
            return True
        else:
            return False

class PlaneWar(gym.Env):
    def __init__(self,n_max_space=1000):
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Discrete(5)
        self.state = 0
        self.screen = None
        self.clock = None
        self.plane = Plane(400, 550)
        self.enemies = []
        self.n_max_space = n_max_space

    def step(self, action):
        reward = 0
        done = False
        # add penalty
        enemy_x_list = [enemy.x for enemy in self.enemies]
        bullet_x_list = [bullet.x for bullet in self.plane.bullets]
        if self.plane.x in enemy_x_list and self.plane.x not in bullet_x_list and action != 2:
            reward-=2
        rampant_enemies = [(enemy.x,enemy.y)  for enemy in self.enemies if enemy.x not in bullet_x_list]
        rampant_enemies_x = np.array([item[0] for item in rampant_enemies])
        rampant_enemies_y = np.array([item[1] for item in rampant_enemies])
        distance = np.abs(np.array(rampant_enemies_y) - 600)
        if len(rampant_enemies) > 0:
            rampant_enemies_x[np.argmin(distance)]
            print('rampant_enemies_x[np.argmin(distance)]',rampant_enemies_x[np.argmin(distance)])
            if (rampant_enemies_x[np.argmin(distance)] < self.plane.x and action == 1) or (rampant_enemies_x[np.argmin(distance)] > self.plane.x and action == 0):
                reward-=2
        # move player plane based on action
        # 往左
        if self.plane.x>5 and action == 0:
            self.plane.x -= 5
        # 往右
        elif self.plane.x<595 and action == 1:
            self.plane.x += 5
        # 射击
        elif action == 2:
            bullet = Bullet(self.plane.x + self.plane.image.get_width() / 2, self.plane.y)
            self.plane.bullets.append(bullet)
        # 其他啥也不做, 不能超出边界
        else:
            print('nothing to do!!')
        # 随机出现一些敌机(0.02的概率)
        if random.random() > 0.98:
            number_of_enemies = random.randint(0, 2)
            for i in range(number_of_enemies):
                x, y = random.randrange(0, 800), 0
                self.enemies.append(Enemy(x, y))
        # update bullets' positions
        for bullet in self.plane.bullets:
            # 当子弹超出边界时消失
            if bullet.y < 0:
                self.plane.bullets.remove(bullet)
                break
            for enemy in self.enemies:
                # calculate reward
                if bullet.check_collision(enemy):
                    self.enemies.remove(enemy)
                    self.plane.bullets.remove(bullet)
                    reward = 5
                    break
            bullet.y -= 5
        # update enemies' positions
        for enemy in self.enemies:
            enemy.y += 2
            # check collision # calculate reward
            if enemy.y > 600 or enemy.check_collision(self.plane):
                done = True
                reward = -1000

        # get new state
        new_bullet_positions = sorted([(bullet.x, bullet.y) for bullet in self.plane.bullets], key=lambda x: x[0])
        new_bullet_x = np.ravel([item[0] for item in new_bullet_positions])
        new_bullet_y = np.ravel([item[1] for item in new_bullet_positions])
        new_bullet_positions = np.ravel(new_bullet_positions)
        print('new_bullet_positions',new_bullet_positions)
        new_enemy_positions = sorted([(enemy.x, enemy.y) for enemy in self.enemies], key=lambda x: x[0])
        new_enemy_x = [item[0] for item in new_enemy_positions]
        new_enemy_y = [item[1] for item in new_enemy_positions]
        new_enemy_positions = np.ravel(new_enemy_positions)
        print('new_enemy_positions',new_enemy_positions)
        new_plane_positions = [(self.plane.x, self.plane.y)]
        new_plane_x = np.ravel([self.plane.x])
        new_plane_x = np.pad(new_plane_x, (0, self.n_max_space - len(new_plane_x)), 'constant', constant_values=(0))
        new_plane_positions = np.ravel(new_plane_positions)
        print('new_plane_positions',new_plane_positions)
        x = [new_bullet_x, new_bullet_y, new_enemy_x, new_enemy_y]
        x = [var[:self.n_max_space] if len(var)>self.n_max_space else np.pad(var, (0, self.n_max_space - len(var)), 'constant', constant_values=(0)) for var in x]
        new_bullet_x, new_bullet_y, new_enemy_x, new_enemy_y = x
        # print('new_bullet_x, new_bullet_y, new_enemy_x, new_enemy_y',new_plane_x, len(new_bullet_x), len(new_bullet_y), len(new_enemy_x), len(new_enemy_y))
        # new_plane_positions = np.pad(new_plane_positions, (0, self.n_max_space - len(new_plane_positions)), 'constant', constant_values=(0))
        # new_bullet_positions = new_bullet_positions[:self.n_max_space] if len(new_bullet_positions)>self.n_max_space else np.pad(new_bullet_positions, (0, self.n_max_space - len(new_bullet_positions)), 'constant', constant_values=(0))
        # new_enemy_positions = new_enemy_positions[:self.n_max_space] if len(new_enemy_positions)>self.n_max_space else np.pad(new_enemy_positions, (0, self.n_max_space - len(new_enemy_positions)), 'constant', constant_values=(0))
        # new_state = (new_plane_positions, new_bullet_positions, new_enemy_positions)
        new_state = (new_plane_x, new_bullet_x, new_bullet_y, new_enemy_x, new_enemy_y)
        

        return np.array(new_state, dtype=np.float32), reward, done, {}

    def reset(self):
        self.state = [[0 for _ in range(self.n_max_space)] for _ in range(self.observation_space.n)]
        # self.state[0][0],self.state[0][1]   = 400, 550
        self.state[0][0] = 400
        self.plane = Plane(400, 550)
        self.enemies = []
        return np.array(self.state, dtype=np.float32)


    def render(self, action=3,reward=0,done=False):
        screen_width = 800
        screen_height = 600

        if self.state is None:
            return None

        x = self.state

        if self.screen is None:
            pygame.init()
            pygame.display.init()
            self.screen = pygame.display.set_mode((screen_width, screen_height))
            pygame.display.set_caption("Airplane War")
        if self.clock is None:
            self.clock = pygame.time.Clock()

        # Draw an image
        image = pygame.image.load("material/background.jpg")
        self.screen.blit(image, (0, 0))

        font = pygame.font.Font(None, 36)
        # render "Score:" text once
        score_label = font.render("Score: ", True, (255, 255, 255))
        score_label_rect = score_label.get_rect()
        score_label_rect.topleft = (10, 10)
        self.screen.blit(score_label, score_label_rect)
        # render score variable
        score = reward
        score_num = font.render(str(score), True, (255, 255, 255))
        score_num_rect = score_num.get_rect()
        score_num_rect.topleft = (score_label_rect.right, 10)
        self.screen.blit(score_num, score_num_rect)

        # Create game over surface
        game_over_surface = pygame.Surface((800, 600))
        game_over_surface.fill((0, 0, 0))

        # Create the game over text
        font = pygame.font.Font(None, 36)
        text = font.render("Game Over", True, (255, 255, 255))
        text_rect = text.get_rect()
        text_rect.center = (400, 300)
        game_over_surface.blit(text, text_rect)

        if done:
            pygame.mixer.music.pause()
            self.screen.blit(game_over_surface, (0, 0))
            self.screen.blit(score_label, score_label_rect)
            finish_text = "Victory!!" if reward == 50  else 'Failure!!'
            self.screen.blit(image, (0, 0))
            self.screen.blit(score_label, score_label_rect)
        else:
            self.plane.draw(self.screen)
            # Check if any enemies have gone past the bottom of the screen
            for enemy in self.enemies:
                    enemy.draw(self.screen)
            for bullet in self.plane.bullets:
                for enemy in self.enemies:
                    if bullet.check_collision(enemy):
                        explosion_image = pygame.image.load("material/boom.png").convert_alpha()
                        explosion_image.set_alpha(200)
                        self.screen.blit(explosion_image, (enemy.x+2, enemy.y+2))
                        break
            
        score_num = font.render(str(score), True, (255, 255, 255))
        score_num_rect = score_num.get_rect()
        score_num_rect.topleft = (score_label_rect.right, 10)
        self.screen.blit(score_num, score_num_rect)
        pygame.display.flip()

        

    def close(self):
        if self.screen is not None:
            pygame.display.quit()
            pygame.quit()
            self.isopen = False