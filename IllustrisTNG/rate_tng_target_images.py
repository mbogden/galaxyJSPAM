import os
import csv
import argparse
from tkinter import Tk, Label, Button, Checkbutton, IntVar, Frame
from tkinter import messagebox
from PIL import Image, ImageTk

class ImageRater:
    def __init__(self, image_dir):
        self.image_dir = image_dir
        self.image_list = self.get_image_list()
        self.current_index = 0
        self.ratings_file = 'ratings.csv'
        self.ratings = {}
        self.load_ratings()
        self.root = Tk()
        self.root.title('Image Rater')
        self.load_rated_images = IntVar(value=1)
        self.setup_gui()

    def get_image_list(self):
        supported_formats = ('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff')
        files = [f for f in os.listdir(self.image_dir) if f.lower().endswith(supported_formats)]
        files.sort()
        return files

    def load_ratings(self):
        if os.path.exists(self.ratings_file):
            with open(self.ratings_file, 'r', newline='', encoding='utf-8') as csvfile:
                reader = csv.reader(csvfile)
                self.ratings = {rows[0]: rows[1] for rows in reader}
        else:
            with open(self.ratings_file, 'w', newline='', encoding='utf-8') as csvfile:
                pass  # Create the file if it doesn't exist

    def setup_gui(self):
        # Image display
        self.image_label = Label(self.root)
        self.image_label.pack()

        # Rating buttons
        button_frame = Frame(self.root)
        button_frame.pack()

        for i in range(6):
            btn = Button(button_frame, text=f'{i} Star', command=lambda rating=i: self.rate_image(rating))
            btn.grid(row=0, column=i)

        # Checkbox
        self.checkbox = Checkbutton(
            self.root,
            text='Load Rated Images',
            variable=self.load_rated_images,
            command=self.on_checkbox_change
        )
        self.checkbox.pack()

        # Navigation buttons
        nav_frame = Frame(self.root)
        nav_frame.pack()

        self.prev_button = Button(nav_frame, text='Previous', command=self.prev_image)
        self.prev_button.grid(row=0, column=0)

        self.next_button = Button(nav_frame, text='Next', command=self.next_image)
        self.next_button.grid(row=0, column=1)

        self.update_image()

    def on_checkbox_change(self):
        self.update_image()

    def update_image(self):
        if not self.image_list:
            messagebox.showinfo("No Images", "No images found in the directory.")
            self.root.quit()
            return

        image_name = self.image_list[self.current_index]
        image_path = os.path.join(self.image_dir, image_name)
        try:
            image = Image.open(image_path)
            image = image.resize((1800, 800), Image.ANTIALIAS)
            self.photo = ImageTk.PhotoImage(image)
            self.image_label.config(image=self.photo)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load image {image_name}.\n{e}")
            self.next_image()

    def rate_image(self, rating):
        image_name = self.image_list[self.current_index]
        self.ratings[image_name] = str(rating)
        with open(self.ratings_file, 'a', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([image_name, rating])
        self.next_image()

    def next_image(self):
        total_images = len(self.image_list)
        for _ in range(total_images):
            self.current_index = (self.current_index + 1) % total_images
            image_name = self.image_list[self.current_index]
            if self.load_rated_images.get() or image_name not in self.ratings:
                self.update_image()
                return
        messagebox.showinfo("No Unrated Images", "No unrated images left.")
        self.root.quit()

    def prev_image(self):
        total_images = len(self.image_list)
        for _ in range(total_images):
            self.current_index = (self.current_index - 1) % total_images
            image_name = self.image_list[self.current_index]
            if self.load_rated_images.get() or image_name not in self.ratings:
                self.update_image()
                return
        messagebox.showinfo("No Unrated Images", "No unrated images left.")
        self.root.quit()

    def run(self):
        self.root.mainloop()

def main():
    parser = argparse.ArgumentParser(description='Image Rating Application')
    parser.add_argument('directory', type=str, help='Directory containing images to review')
    args = parser.parse_args()
    app = ImageRater(args.directory)
    app.run()

if __name__ == '__main__':
    main()
