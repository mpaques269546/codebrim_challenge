

from torchvision import transforms as pth_transforms
from PIL import Image
import sys
from IPython.display import display
import os
import time


def preprocess_image(img_path,  show_img=True):
	transform = pth_transforms.Compose([
        pth_transforms.Resize(256, interpolation=3),
        pth_transforms.CenterCrop(224),
        pth_transforms.ToTensor(),
        pth_transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
	if os.path.isfile(img_path):
		with open(img_path, 'rb') as f:
			img = Image.open(f)
			img = img.convert('RGB')
	if show_img:
		im = img.resize((448, 448))
		display(im)
		#plt.figure()
		#plt.imshow(img)
		#img.show()
	img = transform(img)
	return img.unsqueeze(0)


def make_predictions(model, img_path, labels=[''], show_img=True, show_pred=True):
	t0 = time.time()
	img = preprocess_image(img_path, show_img=show_img)
	preds = model(img)
	t1 = time.time()
	preds = preds.flatten()
	for pred, label in zip(preds, labels):
		progress_bar(pred, label)
	print(f'inference time = {(t1-t0)*10**3:.2f} ms')


def progress_bar(prediction_probability, label):
    assert (prediction_probability>=0.0) and (prediction_probability<=1.0)

    bar_size = 40
    plus_size = int(prediction_probability*bar_size)
    minus_size = bar_size - plus_size
    
    label_name=label
    while len(label_name)<15:
    	label_name = label_name + ' '
    
    sys.stdout.write(label_name)
    sys.stdout.flush()
    for i in range(plus_size):
    	sys.stdout.write("+")
    	sys.stdout.flush()
    for i in range(minus_size):
    	sys.stdout.write(".")
    	sys.stdout.flush()
    sys.stdout.write(f"{prediction_probability*100:>6.2f}% \n")
    sys.stdout.flush()