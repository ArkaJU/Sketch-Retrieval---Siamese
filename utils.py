import cv2
import os

dataset_path = "Sketchy/"
photo_path = os.path.join(dataset_path, 'photo/')
sketch_path = os.path.join(dataset_path, 'sketch/')


def load_img(path):
    img = cv2.imread(path)/255
    return cv2.resize(img, (100,100))


def get_dict():
    
    photo_dictionary = {}

    for category in os.listdir(photo_path):
        category_path = os.path.join(photo_path, category)

        photo_dictionary[category] = os.listdir(category_path)

    sketch_dictionary = {}

    for category in os.listdir(sketch_path):
        category_path = os.path.join(sketch_path, category)

        sketch_dictionary[category] = os.listdir(category_path) 
    
    return photo_dictionary, sketch_dictionary


def get_batch(photo_dictionary, sketch_dictionary):
    
    
    l = []
    p_ = []
    s_ = []

    for _ in range(128): 

        if np.random.uniform() >= 0.5:

            photo_class = np.random.choice(list(photo_dictionary))
            photo = np.random.choice(photo_dictionary[photo_class])
            photo_dictionary[photo_class].remove(photo)
            p = photo_class + '/' + photo

            sketch_class = photo_class
            sketch = np.random.choice(sketch_dictionary[sketch_class])
            sketch_dictionary[sketch_class].remove(sketch)
            s = sketch_class + '/' + sketch
            label = 1

        else:

            x = list(photo_dictionary)
            photo_class = np.random.choice(x)
            photo = np.random.choice(photo_dictionary[photo_class])
            photo_dictionary[photo_class].remove(photo)
            p = photo_class + '/' + photo
            x.remove(photo_class)

            sketch_class = np.random.choice(x)
            sketch = np.random.choice(sketch_dictionary[sketch_class])
            sketch_dictionary[sketch_class].remove(sketch)
            s = sketch_class + '/' + sketch
            label = 0

        p_.append(os.path.join(dataset_path, 'photo/', p))
        s_.append(os.path.join(dataset_path, 'sketch/', s))
        l.append(label)
    
    images = np.array([load_img(i) for i in p_])
    sketches = np.array([load_img(i) for i in s_])
    labels = np.array(l)

    return images, sketches, labels