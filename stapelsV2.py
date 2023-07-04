//python code
import cv2 
import numpy as np 
from stl import mesh 
from PIL import Image 
import pandas as pd 
from sklearn.cluster import KMeans 
import pyvista as pv 
 
image_path = 'Ilona.jpg' 
Kleuren = 5 #Aantal kleuren 
hoogtePrint = 10 #Hoogte van de print 
model_size = 1000 #Kwaliteit 
model_grootte = 200 #Grootste zijde 
 
Printhoogte = model_size/model_grootte*hoogtePrint 
output_file = 'output_model.stl' 
 
def create_3d_model(image_path, model_size, output_file): 
 # Load the image in grayscale 
 image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE) 
 
 # Flip the image horizontally 
 flipped_image = cv2.flip(image, 1) 
 
 original_height, original_width = image.shape 
 aspect_ratio = original_height / original_width 
 
 if original_width > original_height: 
 new_width = model_size 
 new_height = int(new_width * aspect_ratio) 
 else: 
 new_height = model_size 
 new_width = int(new_height / aspect_ratio) 
 
 resized_image = cv2.resize(flipped_image, (new_width, new_height)) 
 
 # Create a meshgrid for the x and y coordinates 
 x, y = np.meshgrid(np.arange(new_width), np.arange(new_height)) 
 
 # Flatten the resized image and normalize pixel intensities 
 z = resized_image.flatten() / np.max(resized_image.flatten()) 
 
 # Apply a scaling factor to increase the peak height 
 z *= Printhoogte 
 
 # Create a 3D mesh for the main model 
 vertices = np.column_stack((x.flatten(), y.flatten(), z)) 
 faces = [] 
 num_rows, num_cols = new_height, new_width 
 for i in range(num_rows - 1): 
 for j in range(num_cols - 1): 
 p1 = i * num_cols + j 
 p2 = p1 + 1 
 p3 = p1 + num_cols 
 p4 = p3 + 1 
 faces.append([p1, p3, p2]) 
 faces.append([p2, p3, p4]) 
 
 # Create the main STL mesh 
 main_mesh = mesh.Mesh(np.zeros(len(faces), dtype=mesh.Mesh.dtype)) 
 for i, f in enumerate(faces): 
 for j in range(3): 
 main_mesh.vectors[i][j] = vertices[f[j], :] 
 
 # Create the bottom layer vertices by duplicating the main model vertices 
 bottom_layer_vertices = vertices.copy() 
 
 # Adjust the height of the bottom layer 
 bottom_layer_vertices[:, 2] = np.min(z) 
 
 # Combine the main model and bottom layer vertices 
 combined_vertices = np.vstack((vertices, bottom_layer_vertices)) 
 
 # Create the bottom layer faces 
 bottom_layer_faces = [[i + len(vertices), i + len(vertices) + 1, i + 1] for i in range(len(vertices) - 1)] 
 bottom_layer_faces.append([len(vertices) - 1, len(vertices), 0]) 
 
 # Create the bottom layer STL mesh 
 bottom_layer_mesh = mesh.Mesh(np.zeros(len(bottom_layer_faces), dtype=mesh.Mesh.dtype)) 
 for i, f in enumerate(bottom_layer_faces): 
 for j in range(3): 
 bottom_layer_mesh.vectors[i][j] = combined_vertices[f[j], :] 
 
 # Create the bottom face by connecting the bottom layer vertices 
 bottom_face = [[i + len(vertices) for i in range(len(vertices))]] 
 
 # Create the bottom face STL mesh 
 bottom_face_mesh = mesh.Mesh(np.zeros(len(bottom_face), dtype=mesh.Mesh.dtype)) 
 for i, f in enumerate(bottom_face): 
 for j in range(3): 
 bottom_face_mesh.vectors[i][j] = combined_vertices[f[j], :] 
 
 # Combine the main model, bottom layer, and bottom face meshes 
 combined_mesh = mesh.Mesh(np.concatenate([main_mesh.data, bottom_layer_mesh.data, bottom_face_mesh.data])) 
 
 # Save the combined STL mesh to a file 
 combined_mesh.save(output_file) 
 
def quantize_image(image, num_colors): 
 # Convert the image to a numpy array 
 img_array = np.array(image) 
 
 # Reshape the array to a 2D matrix of pixels 
 pixels = img_array.reshape(-1, 3) 
 
 # Perform k-means clustering 
 kmeans = KMeans(n_clusters=num_colors, random_state=0).fit(pixels) 
 
 # Get the cluster labels for each pixel 
 labels = kmeans.labels_ 
 
 # Get the RGB values of the cluster centers 
 colors = kmeans.cluster_centers_ 
 
 # Replace each pixel with the RGB value of its cluster center 
 quantized_pixels = colors[labels] 
 
 # Reshape the quantized pixels back to the original shape 
 quantized_img_array = quantized_pixels.reshape(img_array.shape) 
 
 # Convert the numpy array back to PIL Image 
 quantized_image = Image.fromarray(quantized_img_array.astype(np.uint8)) 
 
 return quantized_image 
 
 
def get_grayscale_value(rgb_color): 
 # Calculate the grayscale value from RGB values using the formula 
 gray_value = round(0.299 * rgb_color[0] + 0.587 * rgb_color[1] + 0.114 * rgb_color[2]) 
 
 return gray_value 
 
 
def get_color_palette(image): 
 # Get all unique colors from the image 
 colors = image.getcolors(image.size[0] * image.size[1]) 
 
 # Create a list to store the results 
 results = [] 
 
 # Loop over each color and grayscale value 
 for color in colors: 
 rgb_color = color[1] 
 gray_value = get_grayscale_value(rgb_color) 
 results.append((rgb_color, gray_value)) 
 
 # Create a dataframe from the results 
 results_df = pd.DataFrame(results, columns=["Color", "Grayscale"]) 
 
 # Sort the dataframe based on the grayscale values in ascending order 
 results_df = results_df.sort_values("Grayscale", ascending=True) 
 
 # Calculate Printhoogte % and Snijhoogte 
 gray_values = results_df["Grayscale"].tolist() 
 normalized_colors = [(color / 2.56) for color in gray_values] 
 sorted_colors = sorted(normalized_colors) 
 intermediate_values = [(sorted_colors[i] + sorted_colors[i + 1]) / 2 for i in range(len(sorted_colors) - 1)] 
 intermediate_values.append(100) 
 results_df['SH %'] = intermediate_values 
 results_df['SH deze render'] = results_df['SH %'] * Printhoogte * 0.01 
 results_df['SH echte render'] = results_df['SH %'] * hoogtePrint * 0.01 
 
 return results_df 
create_3d_model(image_path, model_size, output_file) 
 
# Load the STL file 
mesh = pv.read(output_file) 
 
# Calculate the center of the mesh 
center = mesh.center 
 
# Calculate the minimum and maximum z-coordinates 
z_min = np.min(mesh.points[:, 2]) 
z_max = np.max(mesh.points[:, 2]) 
 
layer_height = (z_max - z_min) / 2 
 
# Quantize the image into the specified number of colors 
image = Image.open(image_path) 
quantized_image = quantize_image(image, Kleuren) 
quantized_image.save("output.jpg") 
 
# Get the color palette 
results_df = get_color_palette(quantized_image) 
results_df = results_df.reset_index() 
 
laag1 = results_df.loc[0, 'SH %'] * 0.01 
laag2 = results_df.loc[1, 'SH %'] * 0.01 
laag3 = results_df.loc[2, 'SH %'] * 0.01 
 
# Define the heights for each color 
heights = [z_min, z_min + laag1 * (z_max - z_min), z_min + laag2 * (z_max - z_min), 
 z_min + laag3 * (z_max - z_min), z_max] 
 
# Create an empty list to store the mesh parts 
mesh_parts = [] 
 
# Split the mesh into parts based on the heights 
for i in range(len(heights) - 1): 
 min_height = heights[i] 
 max_height = heights[i + 1] 
 
 # Filter the points that fall within the height range 
 mask = np.logical_and(mesh.points[:, 2] >= min_height, mesh.points[:, 2] < max_height) 
 
 # Extract the part of the mesh based on the mask 
 part = mesh.extract_points(mask) 
 mesh_parts.append(part) 
 
# Create a plotter and add each mesh part with a different color 
plotter = pv.Plotter() 
#colors = ['black', 'grey', 'lightgrey', 'white'] 
colors = [] 
for i in range(Kleuren): 
 grayscale_value = i * 255 / (Kleuren - 1) 
 color = round(grayscale_value) 
 hex_color = '#{:02x}{:02x}{:02x}'.format(color, color, color) 
 colors.append(hex_color) 
colors = [str(color) for color in colors] 
for i in range(len(mesh_parts)): 
 plotter.add_mesh(mesh_parts[i], color=colors[i]) 
 
# Set the background color (optional) 
plotter.set_background("w") 
 
# Start the interactor and display the window 
plotter.show() 
print(results_df) 
print(layer_height) 

 
