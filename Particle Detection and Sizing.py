import cv2
import numpy as np
from tifffile import imread
import os
import pandas as pd
from multiprocessing import Pool

# This Python script processes a directory containing stacks of TIFF images, where each stack represents a series of images.
# Circular particles in each image are detected and measured in terms of size.

# Limits the number of particles counted per sample/stack file
MAX_RADII_PER_STACK = 1500 



def compute_ratio(image, circle):
    '''Takes an image and the parameters of a detected circle, extracts a region around the circle,
     computes the ratio of the detected perimeter pixels to the expected perimeter, and returns this ratio'''

    # Decomposing Circle Parameters
    x, y, r = circle

    # Defining our Region of Interest (ROI)
    y_range = slice(max(0, int(y-r)), min(image.shape[0], int(y+r)))
    x_range = slice(max(0, int(x-r)), min(image.shape[1], int(x+r)))
    region = image[y_range, x_range]

    # Creating a Meshgrid for the region
    y, x = np.ogrid[:region.shape[0], :region.shape[1]]
    y -= (region.shape[0] // 2)
    x -= (region.shape[1] // 2)

    # Define a circular mask and a slightly larger perimeter mask
    mask = x*x + y*y <= r*r
    perimeter_mask = np.logical_and(r*r <= x*x + y*y, x*x + y*y <= (r+1)*(r+1))
    
    # Essentially subtract the mask from the slightly larger perimeter mask, leaving only perimeter pixels
    region_perimeter = np.where(perimeter_mask, region, 0)

    # Counting Detected Perimeter Pixels
    detected_perimeter = np.sum(region_perimeter == 255)
    expected_perimeter = 2 * np.pi * r
            
    # Returns the ratio of the detected particle perimeter to the expected perimeter
    return detected_perimeter / expected_perimeter




def filter_circles_by_ratio(image, circles, threshold=1.00):
    '''Serves to filter out detected particles which are false positives.
    Returns only particles whose ratio of detected particle perimeter to expected perimeter exceeds a chosen threshold.'''
    
    filtered_circles = []
    for circle in circles[0,:]:
        # Ratio of detected particle perimeter to expected perimeter
        ratio = compute_ratio(image, circle)

        # Filter based on the chosen threshold
        if ratio >= threshold:
            filtered_circles.append(circle)
    return np.array([filtered_circles], dtype=np.uint16)




def process_image_stack(stack_path):
    '''Reads a stack of images from a given path. Converts each image to grayscale and applies both Canny Edge Detection and
    the Hough Circle Transform to detect particles. Filters the results based on a number of chosen parameters.
    Displays the images with detected circles overlaid. Breaks out of the loop if the total number of detected particles exceeds MAX_RADII_PER_STACK.'''

    stack = imread(stack_path)
    all_radii = []

    for im in stack:
        # Convert the pixel value range from 16-bit to 8-bit
        scaled_img = cv2.convertScaleAbs(im, alpha=(255.0/65535.0))
        # Convert image to grayscale if not already
        if len(scaled_img.shape) == 3 and scaled_img.shape[2] > 1:
            gray = cv2.cvtColor(scaled_img, cv2.COLOR_BGR2GRAY)
        else:
            gray = scaled_img

        #Thresholding on image to pick out the objects from the background
        #ret, thresh = cv2.threshold(gray, 160, 255, cv2.THRESH_BINARY_INV)

        #Display the grayscale image and the Canny Edge detected image.
        cv2.namedWindow('Gray', cv2.WINDOW_NORMAL)
        cv2.imshow("Gray", gray)

        edges = cv2.Canny(gray, 130, 255)
        cv2.namedWindow('Canny edges', cv2.WINDOW_NORMAL)
        cv2.imshow("Canny edges", edges)


        # Detect circles/particles in each image. Filter the results using the HoughCircles function parameters.
        circles = cv2.HoughCircles(edges, cv2.HOUGH_GRADIENT, 1, minDist=85, param1=12, param2=10, minRadius=5, maxRadius=20000)
        if circles is not None:
            circles = filter_circles_by_ratio(edges, circles, threshold=0.12)  
            radii = [circle[2] for circle in circles[0, :]]
            all_radii.extend(radii)  # Add detected radii to the aggregated list

            for i in circles[0,:]:
                # draw the outer circle
                cv2.circle(gray,(i[0],i[1]),i[2],(255,255,255),2)
                # draw the center of the circle
                cv2.circle(gray,(i[0],i[1]),2,(255,255,255),3)

        # Display the detected circles/particles overlaid on the grayscale image
        cv2.namedWindow('detected circles', cv2.WINDOW_NORMAL)
        cv2.imshow('detected circles', gray)
        #v2.resizeWindow('detected circles', resized_blurimg, height)
        cv2.waitKey()

        # If total number of radii detected exceeds MAX_RADII_COUNT, break out of the loop
        if len(all_radii) >= MAX_RADII_PER_STACK:
            print(f"Reached the maximum limit of {MAX_RADII_PER_STACK} detected radii for stack {stack_path}. Moving to the next stack.")
            break

    return all_radii



# Export to CSV
csv_filename = 'G:\\Wax images test2\\radii_per_stack.csv'

def save_to_csv(stack_path, radii):
    '''Saves the data as it goes by appending to a DataFrame and exporting as a CSV.'''

    # Check if CSV exists
    if os.path.exists(csv_filename):
        df_existing = pd.read_csv(csv_filename)
    else:
        df_existing = pd.DataFrame(columns=['StackPath', 'Radii'])

    # Append new data
    df_new = pd.DataFrame({'StackPath': [stack_path], 'Radii': [radii]})
    df_combined = df_existing.append(df_new, ignore_index=True)

    # Save combined DataFrame to CSV
    df_combined.to_csv(csv_filename, index=False)
    print(f"Radii data exported to {csv_filename}")

def process_file(full_path):
    '''Imports image stacks found in the given directory, acquires the particle radii and exports the results to a CSV.'''
    try:
        # Import an image stack and get the radii of particles detected therein
        radii = process_image_stack(full_path)

        # Save immediately after processing each stack
        save_to_csv(full_path, radii)  
        return (full_path, radii)

    except Exception as e:
        print(f"Error processing {full_path}: {e}")
        return (full_path, [])


def main(directory_path):
    '''Finds .tif files in the given directory path to analyse. Uses multiprocessing to speed up processing.'''
    paths = []
    for dirpath, dirnames, filenames in os.walk(directory_path):
        for filename in filenames:
            if filename.endswith('.tif'):
                full_path = os.path.join(dirpath, filename)
                paths.append(full_path)

    # Counter initialization
    counter = 0
                
    # Using multiprocessing to speed up the image processing
    with Pool(os.cpu_count()) as pool:
        pool.map(process_file, paths)

    print(f"Processed all image stacks in {directory_path}")


# Run the program
if __name__ == "__main__":
    directory_path = 'G:\Wax Images test2'
    main(directory_path)