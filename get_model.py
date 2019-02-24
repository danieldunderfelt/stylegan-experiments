import argparse
import io
import os
import pickle
import config

def main(seed = '', model_name = 'faces', output = config.result_dir):
    model_options = {
        'faces': config.model_faces,
        'celebs': config.model_celebs,
        'cats': config.model_cats,
    }

    model = model_options[model_name]

    print('Extracting model: ' + model_name)

    # Load pre-trained network.
    with open(model, 'rb') as f:
        _G, _D, Gs = pickle.load(f)
        # _G = Instantaneous snapshot of the generator. Mainly useful for resuming a previous training run.
        # _D = Instantaneous snapshot of the discriminator. Mainly useful for resuming a previous training run.
        # Gs = Long-term average of the generator. Yields higher-quality results than the instantaneous snapshot.

    # Save image.
    if output and len(output) > 0:
        result_dir = output 
    else:
        result_dir = config.result_dir

    os.makedirs(config.result_dir, exist_ok=True)
    png_filename = os.path.join(result_dir, str + '_' + model_name + '_' + 'generated.pb')
    Image.fromarray(images[0], 'RGB').save(png_filename)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--output', default='', help='The output directory where you want the file.')
    parser.add_argument('-m', '--model', default='faces', choices=['faces', 'celebs', 'cats'], help='The model that should be used to generate the image.')
    args = parser.parse_args()

    main(model_name=args.model, output=args.output)