from PIL import Image, ImageFilter
import numpy as np
import networkx as nx
import os, time, time

#Finding start and end
def slow_find_startend(image_array, col1, col2):
   #Binary exists so use that instead, if binary is wrong I don't check it, probs should've huh
   if image_array[0][-1][0] == 255 and image_array[0][-1][2] == 0: return quick_find_startend(image_array)

   find = 2
   for i in range(len(image_array)):
      for j in range(len(image_array[i])):
         if np.array_equal(image_array[i][j], col1):
            start = (i, j)
            find -= 1

            image_array = convert_binary(image_array, 0, start)

            if find == 0: 
               save(image_array, "Maze.png")
               return start, end, image_array
            
         elif np.array_equal(image_array[i][j], col2):
            end = (i, j)
            find -= 1

            image_array = convert_binary(image_array, 1, end)

            if find == 0: 
               save(image_array, "Maze.png")
               return start, end, image_array
#Creates colour binary to read the start and end pixels quicker next time
def convert_binary(image_array, pos: int, tuple: tuple[int, int]):
   bin0 = "{0:b}".format(tuple[0])
   bin1 = "{0:b}".format(tuple[1])

   for i in range(len(bin0)):
      image_array[i][-1 -  pos * 2] = np.array([255, 0, 0, 255 - int(bin0[i])])
   for i in range(len(bin1)):
      image_array[i][-1 -  pos * 2 - 1] = np.array([255, 0, 0, 255 - int(bin1[i])])

   return image_array
#Reads the colour binary for the start and end pixels, returns the base2 int
def read_binary(y: int, image_array) -> int:
   binary = ""
   index = 0
   while image_array[index][y][0] == 255 and image_array[index][y][2] == 0:
      if image_array[index][y][3] == 254: binary += "1"
      else: binary += "0"
      index += 1
   return int(binary, 2)
#Binary for start and end exist so read and use them
def quick_find_startend(image_array):

   start = (read_binary(-1, image_array), read_binary(-2, image_array))
   end = (read_binary(-3, image_array), read_binary(-4, image_array)) 

   return start, end, image_array

#Save the image
#Try except isn't really needed anymore but I baby and didn't want it to die, this code may not be parallel but I ran multiple windows of vscode to pretend it was (0 clue if that works)
#Better safe than sorry
def save(image_array, name: str):
   try: Image.fromarray(np.swapaxes(image_array, 0, 1)).save(name)
   except: print(f"{name} not saved")

#Colours
WHITE = np.array([255, 255, 255, 255])
BLACK = np.array([0, 0, 0, 255])
GREEN = np.array([0, 255, 0, 255])
BLUE = np.array([0, 0, 255, 255])
MAGENTA = np.array([255, 0, 255, 255])
AQUA = np.array([0, 255, 255, 255])
RED = np.array([255, 0, 0, 255])



def main():
   #Get the maze image as an array
   image = Image.open("Maze.png")
   image_array = np.swapaxes(np.array(image), 0, 1).copy()

   #Find the route

   start, end, image_array = slow_find_startend(image_array, BLUE, GREEN)
   print(start, end)
   #Search the maze and save the images
   search_around(image_array, start, end)


#Gets the bad apple images
def get_apple_images() -> list[Image.Image]:
   def get_edges(image: Image.Image):
      #Add a pixel of black at top and bottom to be lines if the image is white
      bg1 = Image.new("RGB", (1298,976)) #w+2, h+2
      bg1.paste(image, (0, 1))

      #Set up the image onto a background with the edge filter finding the edges and the outer edges removed
      bg2 = Image.new("RGB", (989,779)) 
      thresh = 200
      fn = lambda x : 255 if x > thresh else 0
      bg1 = bg1.convert('L').point(fn, mode='1').resize((961, 721)).filter(ImageFilter.FIND_EDGES).crop((1, 1, 960, 720))
      bg2.paste(bg1, (30, 60, 989, 779)), #padx, pady
      return bg2  

   name, number, ext = "Frames/frame", 0, ".png"
   images = []
   while os.path.exists(f"{name}{number}{ext}"):
      images.append(get_edges(Image.open(f"{name}{number}{ext}")))
      number += 1
   return images

   
#Checks a square/pixel if they are the right colour, only used once cause it's muuuch slower than checking a Graph node, so only for when making the graph and needs to read the image
def check_square(image_array, pos: tuple[int, int], col):
   return np.array_equal(image_array[pos[0]][pos[1]], col) and np.array_equal(image_array[pos[0] + 1][pos[1]], col) and np.array_equal(image_array[pos[0]][pos[1] + 1], col) and np.array_equal(image_array[pos[0] + 1][pos[1] + 1], col)

#Sets a square/pixel to a color
def set_square(image_array, pos: tuple[int, int], col):
   image_array[pos[0]][pos[1]] = image_array[pos[0] + 1][pos[1]] = image_array[pos[0]][pos[1] + 1] = image_array[pos[0] + 1][pos[1] + 1] = col
   return image_array

#Adds a node with a edge going in allowed directions
def add_node(image_array, G: nx.Graph, pos: tuple[int, int], col):
   if G.has_node((pos[0] // 2, pos[1] // 2)): return image_array, G
   image_array = set_square(image_array, pos, col)
   
   G.add_node((pos[0] // 2, pos[1] // 2))

   positions = [(pos[0] + 2, pos[1]), (pos[0] - 2, pos[1]), (pos[0], pos[1] + 2), (pos[0], pos[1] - 2)]
   for position in positions:
      if G.has_node((position[0] // 2, position[1] // 2)):
         G.add_edge((pos[0] // 2, pos[1] // 2), (position[0] // 2, position[1] // 2))
   
   return image_array, G

def getGraph(image_array, start: tuple[int, int]):
   #Create the graph
   G = nx.Graph()
   #Graph for found nodes so it doesn't keep checking checked nodes, much faster than using a python list cause haha python slow, np would probs also work but hey this is ez
   Gfound = nx.Graph()

   #Positions around start to be checked
   positions = [(start[0] + 2, start[1]), (start[0] - 2, start[1]), (start[0], start[1] + 2), (start[0], start[1] - 2)]

   #Goes through all positions in the maze adding when it finds more
   while len(positions) > 0:
      position = positions.pop(0)

      #If position is part of the maze add it and search around it
      if not Gfound.has_node(position):
         if check_square(image_array, position, WHITE):
            image_array, G = add_node(image_array, G, position, MAGENTA)
            Gfound.add_node(position)

            positions.append((position[0] + 2, position[1]))
            positions.append((position[0] - 2, position[1]))
            positions.append((position[0], position[1] + 2))
            positions.append((position[0], position[1] - 2))
         #If not set it to a wall colour and remove the node
         else:
            image_array = set_square(image_array, position, AQUA)
            Gfound.add_node(position)
   return G, image_array

def pixels(image_array_apple) -> list[tuple[int, int]]:
   apple_pixels = []
   for i in range(len(image_array_apple)):
      for j in range(len(image_array_apple[i])):
         #If pixel is note white
         if np.array_equal(image_array_apple[i][j], WHITE): 
            #No pixels up/left from that pixel
            if not ((i-1, j) or (i, j-1) or (i-1, j-1)) in apple_pixels:
               apple_pixels.append((i, j))
   return apple_pixels

#Searches around the start pixel then creates the images
def search_around(image_array, start: tuple[int, int], end: tuple[int, int]) -> None:
   
   image_array_route_temp = image_array.copy()

   #Working out the maze, overhead so doesn't need to be so fast
   G, image_array = getGraph(image_array, start)


   #Temp image array to be reused later (with route colours)
   image_array_temp = image_array.copy()

   #Add the start and end nodes
   image_array, G = add_node(image_array, G, start, BLUE)
   image_array, G = add_node(image_array, G, end, BLUE)
   save(image_array, "Searched.png")

   print("Clustering grid")
   nx.algorithms.cluster.average_clustering(G)

   startTime = time.perf_counter()
   print("Creating images")
   apple_images = get_apple_images()
   for goes in range(len(apple_images)):
      make_image(goes, apple_images, image_array_temp, image_array_route_temp, start, end, G)
      
   print(time.perf_counter() - startTime)

def make_image(goes: int, apple_images: list[Image.Image], image_array_temp, image_array_route_temp, start: tuple[int, int], end: tuple[int, int], G: nx.Graph) -> None:
#Get an image to draw the route to
   image_array = image_array_temp.copy()
   image_array_route = image_array_route_temp.copy()

   #Get the bad apple image and find the black pixels
   image = apple_images[goes].convert("RGBA")
   image_array_apple = np.swapaxes(np.array(image), 0, 1).copy()
   apple_pixels = pixels(image_array_apple)
   #Total of black pixels found
   total = len(apple_pixels)

   #Start at the start (since it moves from the last position)
   last_pos = (start[0] // 2, start[1] // 2)
   print(f"{goes + 1}/{len(apple_images)}")
   print(f"{goes + 1}/{len(apple_images)} - 0/{total}")

   GnotRouted = G.copy()

   while len(apple_pixels) > 0:
      #Find the closest unfound pixel that needs to be and go to that
      #Makes the image look nicer and is faster since you are going to a node that might be closer, though not guarenteed since the maze could make it very far away
      index = closest_node(last_pos, apple_pixels)

      #Point you're travelling to
      position = apple_pixels.pop(index)

      if GnotRouted.has_node(position):
         route = []
         for point in nx.algorithms.astar_path(G, last_pos, position):
            #If a point is already found in the route, start from there
            if not GnotRouted.has_node(point):
               route = []
            #Add to the route
            route.append(point)
         #Draw out the route for the final image and set explored
         for point in route:
            image_array_route = set_square(image_array_route, (point[0] * 2, point[1] * 2), RED)
            if GnotRouted.has_node(point):
               GnotRouted.remove_node(point)
         #Set the position to move from to the position
         last_pos = position
      
      #Go to points around point to increase clarity, threshold is how thicc you want the lines to be, I did 2
      #Too much and it's uglier than normal, too few and u can't tell so well, but ofc this slows it down
      thresh, positions = 2, []
      for i in range(2, thresh + 2):
         positions.append((position[0] + i, position[1]))
         positions.append((position[0], position[1] + i))
         positions.append((position[0] + i, position[1] + i))
         positions.append((position[0] - i, position[1]))
         positions.append((position[0], position[1] - i))
         positions.append((position[0] - i, position[1] - i))
      for _position in positions:
         if GnotRouted.has_node(_position):
            route = []
            for point in nx.algorithms.astar_path(G, last_pos, _position):
               #If a point is already found in the route, start from there
               #Since the node is already found it won't make any disconnected paths but will look a little nicer
               if not GnotRouted.has_node(point):
                  route = []
               #Add to the route
               route.append(point)
            if len(route) < (0.5 * (len(image_array) + len(image_array[0]))) / 41:
               #Draw out the route for the final image and set explored
               for point in route:
                  image_array_route = set_square(image_array_route, (point[0] * 2, point[1] * 2), RED)
                  if GnotRouted.has_node(point):
                     GnotRouted.remove_node(point)
               #Set the position to move from to the position
               last_pos = _position

      print(f"{goes + 1}/{len(apple_images)} - {total - len(apple_pixels)}/{total}")


   #Last node to the end
   for point in nx.algorithms.astar_path(G, last_pos, (end[0]//2,end[1]//2)):
      image_array_route = set_square(image_array_route, (point[0] * 2, point[1] * 2), RED)

   #Save the frame then lets gogogo again
   save(image_array_route, f"Result/{goes}.png")

#Stole this from a user called Jaime on a old stackextchange thing, ty Jaime
def closest_node(node: tuple[int, int], nodes: list[tuple[int, int]]) -> int:
   nodes = np.asarray(nodes)
   dist_2 = np.sum((nodes - node)**2, axis=1)
   return np.argmin(dist_2)

padx = 30
pady = 60

if __name__ == "__main__":
   main()