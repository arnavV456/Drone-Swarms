import pygame
import numpy as np
import random

# -----------------------------
# PARAMETERS
# -----------------------------
WIDTH, HEIGHT = 1000, 700
NUM_BOIDS = 20

MAX_SPEED = 1
MAX_FORCE = 0.05

NEIGHBOR_RADIUS = 60
SEPARATION_RADIUS = 40
OBSTACLE_RADIUS = 80

# Boundary
BOUNDARY_MARGIN = 80
W_BOUNDARY = 1.5

# Weights
W_SEP = 3.0
W_ALIGN = 1.0
W_COH = 1.0
W_GOAL = 1.0
W_OBS = 20

# -----------------------------
# TARGET
# -----------------------------
TARGET = np.array([WIDTH // 2, HEIGHT // 2])

# -----------------------------
# OBSTACLES (x, y, radius)
# -----------------------------
OBSTACLES = [
    (300, 300, 40),
    (700, 400, 50),
    (500, 150, 30)
]

# -----------------------------
# UTILITY
# -----------------------------
def limit(vector, max_val):
    mag = np.linalg.norm(vector)
    if mag > max_val:
        return (vector / mag) * max_val
    return vector


# -----------------------------
# BOID CLASS
# -----------------------------
class Boid:
    def __init__(self, position=None):
        if position is None:
            self.position = np.array([
                random.uniform(0, WIDTH),
                random.uniform(0, HEIGHT)
            ])
        else:
            self.position = position

        angle = random.uniform(0, 2*np.pi)
        self.velocity = np.array([np.cos(angle), np.sin(angle)])

    def update(self):
        self.velocity = limit(self.velocity, MAX_SPEED)
        self.position += self.velocity

    def apply_force(self, force):
        self.velocity += force

    def flock(self, boids):
        total = 0
        sep = np.zeros(2)
        align = np.zeros(2)
        coh = np.zeros(2)

        for other in boids:
            if other is self:
                continue

            dist = np.linalg.norm(other.position - self.position)

            if dist < NEIGHBOR_RADIUS:
                total += 1

                # ALIGNMENT
                align += other.velocity

                # COHESION
                coh += other.position

                # SEPARATION
                if dist < SEPARATION_RADIUS and dist > 0:
                    diff = self.position - other.position
                    diff /= dist
                    sep += diff

        if total > 0:
            # ALIGNMENT
            align /= total
            align = align - self.velocity
            align = limit(align, MAX_FORCE)

            # COHESION
            coh /= total
            coh = coh - self.position
            coh = limit(coh, MAX_FORCE)

            # SEPARATION
            sep /= total
            sep = limit(sep, MAX_FORCE)

            self.apply_force(W_SEP * sep)
            self.apply_force(W_ALIGN * align)
            self.apply_force(W_COH * coh)

        # -----------------------------
        # GOAL FORCE
        # -----------------------------
        goal_force = TARGET - self.position
        goal_force = limit(goal_force, MAX_FORCE)
        self.apply_force(W_GOAL * goal_force)

        # -----------------------------
        # OBSTACLE AVOIDANCE
        # -----------------------------
        obs_force = np.zeros(2)

        for ox, oy, r in OBSTACLES:
            obstacle_pos = np.array([ox, oy])
            dist = np.linalg.norm(self.position - obstacle_pos)

            if dist < (r + OBSTACLE_RADIUS):
                diff = self.position - obstacle_pos
                if dist != 0:
                    diff /= dist
                obs_force += diff / max(dist, 1)

        obs_force = limit(obs_force, MAX_FORCE)
        self.apply_force(W_OBS * obs_force)

        # -----------------------------
        # BOUNDARY AVOIDANCE (NEW)
        # -----------------------------
        boundary_force = np.zeros(2)

        # Left wall
        if self.position[0] < BOUNDARY_MARGIN:
            boundary_force[0] = 1 / max(self.position[0], 1)

        # Right wall
        elif self.position[0] > WIDTH - BOUNDARY_MARGIN:
            boundary_force[0] = -1 / max(WIDTH - self.position[0], 1)

        # Top wall
        if self.position[1] < BOUNDARY_MARGIN:
            boundary_force[1] = 1 / max(self.position[1], 1)

        # Bottom wall
        elif self.position[1] > HEIGHT - BOUNDARY_MARGIN:
            boundary_force[1] = -1 / max(HEIGHT - self.position[1], 1)

        boundary_force = limit(boundary_force, MAX_FORCE)
        self.apply_force(W_BOUNDARY * boundary_force)

    def draw(self, screen):
        pygame.draw.circle(screen, (255, 255, 255), self.position.astype(int), 3)


# -----------------------------
# MAIN
# -----------------------------
def main():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Drone Swarm Simulator (Boids + Goal + Obstacles + Boundary)")

    clock = pygame.time.Clock()
    boids = []
    boids = []

    for i in range(NUM_BOIDS):
        if i < NUM_BOIDS // 2:
            # LEFT SIDE SPAWN
            position = np.array([
                random.uniform(0, WIDTH * 0.25),
                random.uniform(0, HEIGHT)
            ])
        else:
            # RIGHT SIDE SPAWN
            position = np.array([
                random.uniform(WIDTH * 0.75, WIDTH),
                random.uniform(0, HEIGHT)
            ])

        boids.append(Boid(position))
    global TARGET

    running = True
    while running:
        clock.tick(60)
        screen.fill((0, 0, 0))

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            # Move target with mouse
            if event.type == pygame.MOUSEBUTTONDOWN:
                TARGET = np.array(pygame.mouse.get_pos())

        # Draw target
        pygame.draw.circle(screen, (255, 0, 0), TARGET.astype(int), 6)

        # Draw obstacles
        for ox, oy, r in OBSTACLES:
            pygame.draw.circle(screen, (0, 0, 255), (ox, oy), r)

        # Update boids
        for boid in boids:
            boid.flock(boids)
            boid.update()
            boid.draw(screen)

        pygame.display.flip()

    pygame.quit()


if __name__ == "__main__":
    main()