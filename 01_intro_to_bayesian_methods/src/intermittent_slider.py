import tkanvas
import argparse
from tkanvas import TKanvas
import numpy as np
import numpy.ma as ma
import time
import slider_filter
import sys


class Box:
    def __init__(self, name, left, right, prior, color):
        self.name = name
        self.left = left
        self.right = right
        self.prior = prior
        self.color = color


def clamp(val, minimum=0, maximum=255):
    if val < minimum:
        return minimum
    if val > maximum:
        return maximum
    return val


def colorscale(hexstr, scalefactor):
    """
    Scales a hex string by ``scalefactor``. Returns scaled hex string.

    To darken the color, use a float value between 0 and 1.
    To brighten the color, use a float value greater than 1.

    >>> colorscale("#DF3C3C", .5)
    #6F1E1E
    >>> colorscale("#52D24F", 1.6)
    #83FF7E
    >>> colorscale("#4F75D2", 1)
    #4F75D2
    """

    hexstr = hexstr.strip("#")

    if scalefactor < 0 or len(hexstr) != 6:
        return hexstr

    r, g, b = int(hexstr[:2], 16), int(hexstr[2:4], 16), int(hexstr[4:], 16)

    r = clamp(r * scalefactor)
    g = clamp(g * scalefactor)
    b = clamp(b * scalefactor)

    return "#%02x%02x%02x" % (int(r), int(g), int(b))


def box_probability(particles, weights, boxes):
    xs = particles[:, 0]
    probs = []
    for box in boxes:
        # compute weight in each box
        posterior = np.sum(
            weights[np.logical_and(xs > box.left, xs < box.right)] * box.prior
        )
        probs.append(posterior)
    probs = np.array(probs) + 1e-6

    # normalise probabilities and return
    return probs / np.sum(probs)


class SliderDemo(object):
    def __init__(self, boxes, args):
        self.boxes = boxes
        self.screen_size = 1000
        self.slider_height = 150
        self.canvas = TKanvas(
            draw_fn=self.draw,
            event_fn=self.event,
            quit_fn=self.quit,
            w=self.screen_size,
            h=self.slider_height,
        )
        self.n_particles = 200
        self.particles = slider_filter.prior(self.n_particles)
        # self.canvas.root.config(cursor='none')
        self.show_particles = args.particles

        self.last_x = None
        self.last_t = time.time()
        self.dx = 0
        self.x = 0

    def quit(self, src):
        return

    def event(self, src, event_type, event):

        pass

    def update_filter(self):
        observation = [self.x if self.observed else np.nan, np.abs(self.dx)]
        observation = ma.masked_invalid(observation)
        self.particles, self.weights = slider_filter.filter_step(
            self.particles, observed=observation, dt=self.dt, prior_rate=0.00005
        )
        self.weights = self.weights.filled(0)

    def update_state(self, screen_x):
        t = time.time()
        self.x = screen_x / self.screen_size

        if self.last_x != None:
            dx = self.x - self.last_x
            dt = t - self.last_t
            self.dx = dx / dt
        else:
            dt = 1 / 60.0

        self.last_x = self.x
        self.last_t = t
        self.dt = dt
        self.observed = np.random.uniform(0, 1) < 0.05

    def draw(self, src):
        screen_x = src.mouse_x
        self.update_state(screen_x)
        self.update_filter()
        src.clear()

        box_weights = box_probability(self.particles, self.weights, self.boxes)

        for box, box_w in zip(self.boxes, box_weights):
            screen_left = box.left * self.screen_size
            screen_right = box.right * self.screen_size
            # show fading boxes if in particle mode
            if self.show_particles:
                scaled_color = colorscale(box.color, box_w)
                src.rectangle(
                    screen_left,
                    0,
                    screen_right,
                    self.slider_height,
                    fill=scaled_color,
                    outline=box.color,
                )
            else:
                src.rectangle(
                    screen_left,
                    0,
                    screen_right,
                    self.slider_height,
                    fill=box.color,
                    outline=box.color,
                )

        if self.show_particles:
            for particle, w in zip(self.particles, self.weights):
                x = particle[0] * self.screen_size
                x = x if np.isfinite(x) else 0
                w = w if np.isfinite(w) else 0
                src.circle(x, self.slider_height // 2, np.sqrt(w) * 20 + 1, fill="grey")

            expected_pos = slider_filter.expected_position(self.particles, self.weights)
            expected_var = slider_filter.variance_position(self.particles, self.weights)

            src.circle(
                expected_pos[0] * self.screen_size,
                self.slider_height // 2,
                5,
                fill="blue",
            )
            src.line(
                expected_pos[0] * self.screen_size,
                0,
                expected_pos[0] * self.screen_size,
                self.slider_height,
                fill="blue",
            )

            src.line(
                (expected_pos[0] - np.sqrt(expected_var[0])) * self.screen_size,
                self.slider_height // 2,
                (expected_pos[0] + np.sqrt(expected_var[0])) * self.screen_size,
                self.slider_height // 2,
                fill="blue",
            )

        if self.observed:
            src.line(screen_x, 0, screen_x, self.slider_height, fill="white", width=5)
            src.line(
                screen_x + 5, 0, screen_x + 5, self.slider_height, fill="black", width=5
            )


def create_slider():
    boxes = [
        Box("A", 0.0, 0.1, prior=0.25, color="#ff7200"),
        Box("X", 0.1, 0.9, prior=0.25, color="#3e3e3f"),
        Box("B", 0.9, 1.0, prior=0.5, color="#086375"),
    ]

    parser = argparse.ArgumentParser(description="Slider demo")
    parser.add_argument("--particles", action="store_true")

    args = parser.parse_args()
    print(args)

    slider = SliderDemo(boxes, args)
    return slider


if __name__ == "__main__":

    slider = create_slider()
    slider.canvas.root.mainloop()

