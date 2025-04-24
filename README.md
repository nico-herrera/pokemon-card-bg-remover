# Pokemon Card Background Removal

This model uses the IS-Net architecture from the DIS (Dichotomous Image Segmentation) paper to remove backgrounds from Pokemon card images with a special focus on preserving the card's colored details.

## Model Details

- **Model**: IS-Net (DIS)
- **Weights**: isnet-general-use.pth
- **Primary capability**: High-quality background removal with precise edge detection

## Features

- Removes backgrounds from Pokemon cards while preserving card edges
- Special color preservation mode for maintaining vibrant card borders and artwork
- Edge smoothing for clean, professional-looking results
- Optimized for trading cards with detailed artwork

## Example Input/Output

_Input_: A Pokemon card with background
_Output_: The same Pokemon card with transparent background (PNG format)

## Parameters

- **image**: The input image containing a Pokemon card
- **color_threshold** (default: 0.5): Threshold for identifying colored regions (0.0-1.0)
- **preserve_colors** (default: true): Special handling to preserve colored areas of the card
- **edge_smooth** (default: 2): Amount of edge smoothing (0-5)

## Usage

```python
import replicate

output = replicate.run(
    "your-username/pokemon-card-background-remover",
    input={
        "image": open("pokemon_card.jpg", "rb"),
        "color_threshold": 0.5,
        "preserve_colors": True,
        "edge_smooth": 2
    }
)

# Save the output image
from urllib.request import urlretrieve
urlretrieve(output, "pokemon_card_no_bg.png")
```

## How It Works

1. The model detects the Pokemon card in the image
2. It applies the IS-Net segmentation model to separate the card from the background
3. Special processing is applied to better preserve colored regions of the card
4. Edge smoothing creates a professional-looking final result

## Credits

This model uses the IS-Net architecture from the ["Highly Accurate Dichotomous Image Segmentation"](https://github.com/xuebinqin/DIS) paper (ECCV 2022).

## Deployment Instructions

### Local Testing

1. Install Cog:

   ```bash
   sudo curl -o /usr/local/bin/cog -L https://github.com/replicate/cog/releases/latest/download/cog_`uname -s`_`uname -m`
   sudo chmod +x /usr/local/bin/cog
   ```

2. Run the model locally:
   ```bash
   cog predict -i image=@pokemon_card.jpg
   ```

### Deploying to Replicate

1. Build the model:

   ```bash
   cog build
   ```

2. Push to Replicate:

   ```bash
   cog push replicate {username}/{model-name}
   ```

3. Your model will be available at `replicate.com/{username}/{model-name}`
