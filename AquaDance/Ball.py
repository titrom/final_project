class Ball:
    def __init__(self, x, vx, y, vy, mass):
        self.x = x
        self.vx = vx
        self.y = y
        self.vy = vy
        self.mass = mass

    def set_x(self, x):
        self.x = x

    def get_x(self):
        return self.x

    def set_vx(self, vx):
        self.vx = vx

    def get_vx(self):
        return self.vx

    def set_y(self, y):
        self.y = y

    def get_y(self):
        return self.y

    def set_vy(self, vy):
        self.vy = vy

    def get_vy(self):
        return self.vy

    def get_mass(self):
        return self.mass