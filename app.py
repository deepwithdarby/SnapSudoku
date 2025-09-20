import gradio as gr
import numpy as np
import cv2
import pickle
import os
import random
import time
from random import shuffle

# --- helpers.py ---
class Helpers(object):
    def __init__(self):
        pass

    def thresholdify(self, image):
        image = cv2.adaptiveThreshold(
            image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
        return image

    def largestContour(self, image):
        contours, h = cv2.findContours(
            image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return max(contours, key=cv2.contourArea)

    def largest4SideContour(self, image):
        contours, h = cv2.findContours(
            image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        all_contours = sorted(contours, key=cv2.contourArea, reverse=True)
        for i in all_contours:
            if len(self.approx(i)) == 4:
                return i
        return None

    def cut_out_sudoku_puzzle(self, image, contour):
        x, y, w, h = cv2.boundingRect(contour)
        return image[y:y + h, x:x + w]

    def approx(self, cnt):
        peri = cv2.arcLength(cnt, True)
        return cv2.approxPolyDP(cnt, 0.02 * peri, True)

    def get_rectangle_corners(self, contour):
        return np.array([i[0] for i in contour], dtype=np.float32)

    def warp_perspective(self, corners, image):
        rect = np.zeros((4, 2), dtype=np.float32)
        s = corners.sum(axis=1)
        rect[0] = corners[np.argmin(s)]
        rect[2] = corners[np.argmax(s)]

        diff = np.diff(corners, axis=1)
        rect[1] = corners[np.argmin(diff)]
        rect[3] = corners[np.argmax(diff)]

        (tl, tr, br, bl) = rect

        widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
        widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))

        heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
        heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))

        side = max(int(widthA), int(widthB), int(heightA), int(heightB))

        dst = np.array([
            [0, 0],
            [side - 1, 0],
            [side - 1, side - 1],
            [0, side - 1]], dtype="float32")

        M = cv2.getPerspectiveTransform(rect, dst)
        return cv2.warpPerspective(image, M, (side, side))

# --- cells.py ---
class Cells(object):
    def __init__(self, sudoku_image):
        self.sudoku_image = sudoku_image
        self.side = sudoku_image.shape[0] / 9
        self.cells = self.split_into_cells()

    def split_into_cells(self):
        cells = []
        for i in range(9):
            row = []
            for j in range(9):
                p1 = (int(i * self.side), int(j * self.side))
                p2 = (int((i + 1) * self.side), int((j + 1) * self.side))
                cell = self.sudoku_image[p1[1]:p2[1], p1[0]:p2[0]]
                cell = self.process_cell(cell)
                row.append(cell)
            cells.append(row)
        return cells

    def process_cell(self, cell):
        cell = self.center_the_digit(cell)
        if cell is None:
            return None
        cell = cv2.resize(cell, (28, 28), interpolation=cv2.INTER_CUBIC)
        return cell

    def center_the_digit(self, cell):
        contours, h = cv2.findContours(
            cell, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return np.zeros(cell.shape)

        contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(contour)

        if w > h:
            y = y - (w - h) // 2
            h = w
        if h > w:
            x = x - (h - w) // 2
            w = h

        new_cell = cv2.bitwise_not(cell)
        digit = new_cell[y: y + h, x: x + w]

        side = digit.shape[0]

        if side > (self.side * 0.4):
            new_side = int(self.side - 10)
            digit = cv2.resize(digit, (new_side, new_side),
                               interpolation=cv2.INTER_AREA)

            new_cell = np.zeros((int(self.side), int(self.side)))
            a = (int(self.side) - new_side) // 2
            new_cell[a:a + new_side, a:a + new_side] = digit
            return new_cell
        return np.zeros(cell.shape)

# --- sudokuExtractor.py ---
class Extractor(object):
    def __init__(self, image):
        self.helpers = Helpers()
        self.image = image
        self.preprocess()
        sudoku = self.cropSudoku()
        sudoku = self.straighten(sudoku)
        self.cells = Cells(sudoku).cells

    def preprocess(self):
        self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        self.image = self.helpers.thresholdify(self.image)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
        self.image = cv2.morphologyEx(self.image, cv2.MORPH_CLOSE, kernel)

    def cropSudoku(self):
        contour = self.helpers.largestContour(self.image.copy())
        sudoku = self.helpers.cut_out_sudoku_puzzle(self.image.copy(), contour)
        return sudoku

    def straighten(self, sudoku):
        largest = self.helpers.largest4SideContour(sudoku.copy())
        app = self.helpers.approx(largest)
        corners = self.helpers.get_rectangle_corners(app)
        sudoku = self.helpers.warp_perspective(corners, sudoku)
        return sudoku

# --- train.py ---
def sigmoid(z):
    return (1.0 / (1.0 + np.exp(-z)))

def sigmoid_prime(z):
    return sigmoid(z) * (1 - sigmoid(z))

class CrossEntropyCost(object):
    @staticmethod
    def cost(a, y):
        return np.nan_to_num(-y * np.log(a) - (1 - y) * np.log(1 - a))

    @staticmethod
    def delta(z, a, y):
        return (a - y)

class NeuralNetwork(object):
    def __init__(self, sizes=None, cost=CrossEntropyCost, customValues=None):
        if not customValues:
            self.layers = len(sizes)
            self.sizes = sizes
            self.biases = [np.random.randn(x, 1) for x in sizes[1:]]
            self.wts = [np.random.randn(y, x)
                        for x, y in zip(sizes[:-1], sizes[1:])]
        else:
            self.sizes, self.biases, self.wts = customValues
            self.layers = len(self.sizes)
        self.cost = cost

    def feedforward(self, inputs):
        res = inputs
        for w, b in zip(self.wts, self.biases):
            res = sigmoid(np.dot(w, res) + b)
        return res

# --- sudopy.py ---
def cross(A, B):
    return [a+b for a in A for b in B]

digits   = '123456789'
rows     = 'ABCDEFGHI'
cols     = digits
squares  = cross(rows, cols)
unitlist = ([cross(rows, c) for c in cols] +
            [cross(r, cols) for r in rows] +
            [cross(rs, cs) for rs in ('ABC','DEF','GHI') for cs in ('123','456','789')])
units = dict((s, [u for u in unitlist if s in u])
             for s in squares)
peers = dict((s, set(sum(units[s],[]))-set([s]))
             for s in squares)

def parse_grid(grid):
    values = dict((s, digits) for s in squares)
    for s,d in grid_values(grid).items():
        if d in digits and not assign(values, s, d):
            return False
    return values

def grid_values(grid):
    chars = [c for c in grid if c in digits or c in '0.']
    assert len(chars) == 81
    return dict(zip(squares, chars))

def assign(values, s, d):
    other_values = values[s].replace(d, '')
    if all(eliminate(values, s, d2) for d2 in other_values):
        return values
    else:
        return False

def eliminate(values, s, d):
    if d not in values[s]:
        return values
    values[s] = values[s].replace(d,'')
    if len(values[s]) == 0:
        return False
    elif len(values[s]) == 1:
        d2 = values[s]
        if not all(eliminate(values, s2, d2) for s2 in peers[s]):
            return False
    for u in units[s]:
        dplaces = [s for s in u if d in values[s]]
        if len(dplaces) == 0:
            return False
        elif len(dplaces) == 1:
            if not assign(values, dplaces[0], d):
                return False
    return values

def solve(grid): return search(parse_grid(grid))

def search(values):
    if values is False:
        return False
    if all(len(values[s]) == 1 for s in squares):
        return values
    n,s = min((len(values[s]), s) for s in squares if len(values[s]) > 1)
    return some(search(assign(values.copy(), s, d))
                for d in values[s])

def some(seq):
    for e in seq:
        if e: return e
    return False

# --- sudoku_str.py ---
class SudokuStr(object):
    def __init__(self, sudoku):
        self.s = self.sudoku_to_str(sudoku)

    @staticmethod
    def sudoku_to_str(sudoku):
        s = ''
        if isinstance(sudoku, str):
            s = sudoku
        elif isinstance(sudoku, (list, tuple)):
            if len(sudoku) == 9:
                s = ''.join(''.join(row for row in col) for col in sudoku)
            elif len(sudoku) == 81:
                s = ''.join(sudoku)
        return s.replace(' ', '.').replace('0', '.').replace('_', '.')

    def __repr__(self):
        return "{}('{}')".format(self.__class__.__name__, self.s)

    def __str__(self):
        return self.sudoku_board()

    @staticmethod
    def border_line():
        return ('-' * 7).join('|' * 4)

    @staticmethod
    def get_fmt(i):
        return '{}' if i % 3 else '| {}'

    @classmethod
    def sudoku_line(cls, i, line):
        s = '' if i % 3 else cls.border_line() + '\n'
        return s + ' '.join(cls.get_fmt(i).format(x if x != '0' else '_')
            for i, x in enumerate(line)) + ' |'

    def board_rows(self):
        for i in range(9):
            yield self.s[i*9:(i+1)*9]

    def sudoku_board(self):
        solved_grid = solve(self.s)
        if solved_grid:
            self.s = "".join(solved_grid[s] for s in squares)
        return '\n'.join(self.sudoku_line(i, line) for i, line
            in enumerate(self.board_rows())) + '\n' + self.border_line()

    def solve(self):
        parsed_grid = parse_grid(self.s)
        if not parsed_grid:
            raise ValueError('Sudoku puzzle is not solvable.\n> ' + self.s)

        solution = search(parsed_grid)
        if not solution:
            raise ValueError('Sudoku puzzle is not solvable.\n> ' + self.s)

        self.s = "".join(solution[s] for s in squares)
        return self


# --- Main Application Logic ---
def create_net(rel_path):
    with open(os.getcwd() + rel_path, 'rb') as in_file:
        u = pickle._Unpickler(in_file)
        u.encoding = 'latin1'
        sizes, biases, wts = u.load()
    return NeuralNetwork(customValues=(sizes, biases, wts))

def get_cells_from_image(image):
    net = create_net(rel_path='/networks/net')
    extractor = Extractor(image)
    for row in extractor.cells:
        for cell in row:
            if cell is None:
                yield '.'
                continue
            x = net.feedforward(np.reshape(cell, (784, 1)))
            x[0] = 0
            digit = np.argmax(x)
            if float(list(x[digit])[0]) / sum(x) > 0.8:
                 yield str(digit)
            else:
                yield '.'

def solve_sudoku(image):
    if image is None:
        return "Please upload an image."
    try:
        grid = ''.join(cell for cell in get_cells_from_image(image))
        s = SudokuStr(grid)
        return str(s.solve())
    except Exception as e:
        return "Error: Could not solve Sudoku. Please try another image."

# --- Gradio Interface ---
iface = gr.Interface(
    fn=solve_sudoku,
    inputs=gr.Image(type="numpy"),
    outputs="text",
    title="Sudoku Solver",
    description="Upload an image of a Sudoku puzzle and see the solution."
)

if __name__ == "__main__":
    iface.launch()
