import time

import torch
import torch.nn.functional as F

from torch import nn, optim
from torchvision import datasets, transforms

from utils import timer, predict_args, model_categories, process_image, load_checkpoint


def predict(args):
    checkpoint = args.checkpoint
    input_image = args.input_image
    top_k = args.top_k
    category_names = args.category_names
    device = 'cuda' if args.gpu else 'cpu'    
    
    print(f"Preparing top {top_k} predictions with checkpoint {checkpoint} with {device} device.")
    
    load_time = time.time()
    model = load_checkpoint(checkpoint, device)
    image_np = process_image(input_image)
    
    top_ps, top_classes = evaluate(model, image_np, top_k, category_names, device)
    
    print(f'Predictions for image {input_image}')
    print('Prob\tClass')
    for pred in list(zip(top_ps.tolist()[0], top_classes)):
        print('{:.2f}%\t[{}] {}'.format(pred[0] * 100, pred[1][0], pred[1][1].capitalize()))
    
@timer
def evaluate(model, image_np, top_k, category_names, device):
    image_torch = torch.from_numpy(image_np)
    image_torch.unsqueeze_(0)
    
    with torch.no_grad():
        img = image_torch.to(device)
        log_ps = model.forward(img.float())
        
    ps = torch.exp(log_ps)
    top_ps, top_class_ids = ps.topk(top_k)
    
    categories = model_categories(category_names, model.class_to_idx.items())
    top_classes = [categories[cat] for cat in top_class_ids.tolist()[0]]

    return top_ps, top_classes

if __name__ == '__main__':
    parser = predict_args()
    args = parser.parse_args()
    
    predict(args)