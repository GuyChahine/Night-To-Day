from PIL import Image
import numpy as np
import cv2
from keras.models import load_model
from keras_contrib.layers.normalization.instancenormalization import InstanceNormalization

class Model():
    
    def __init__(self, models_path):
        custom_object = {"InstanceNormalization": InstanceNormalization}
        # Load Models
        self.gen_AB = load_model(models_path + "gen_AB_e59.keras", custom_objects=custom_object)
        self.gen_BA = load_model(models_path + "gen_BA_e77.keras", custom_objects=custom_object)
        
        """# Load EDSR_x4
        self.edsr_x4 = cv2.dnn_superres.DnnSuperResImpl_create()
        self.edsr_x4.readModel(models_path + "EDSR_x4.pb")
        self.edsr_x4.setModel('edsr',4)"""
        
        """# Sharp kernel
        self.kernel = np.array([
            [-1,-1,-1],
            [-1,9,-1],
            [-1,-1,-1],
        ])"""
        
    def before_processing(self, image, AB):
        """if AB:
            image = np.where(image < 30, image+30, image) # Increase Contrast"""
        base_shape = image.shape
        image = cv2.resize(image, (512,512)).astype(np.float32)/255
        assert image.shape == (512,512,3), f"Image shape = {image.shape}, model need (512,512,3)"
        return image.reshape(1,512,512,3), base_shape
    
    def after_processing(self, image, base_shape):
        image = np.clip(image, 0, 1)
        image = cv2.resize(image, (base_shape[1], base_shape[0]))
        image = (image - image.min())/(image.max() - image.min())
        """image = cv2.filter2D(image, -1, self.kernel)
        image = self.edsr_x4.upsample(image)"""
        image = (image*255).astype(np.uint8)
        return Image.fromarray(image)
    
    def predict_AB(self, image):
        print("predAB")
        image, base_shape = np.array(self.before_processing(image, True))
        print("before")
        prediction = self.gen_AB.predict(image, verbose=0)
        print("after")
        return self.after_processing(prediction[0], base_shape)
    
    def predict_BA(self, image):
        image, base_shape = np.array(self.before_processing(image, False))
        prediction = self.gen_BA.predict(image, verbose=0)
        return self.after_processing(prediction[0], base_shape)