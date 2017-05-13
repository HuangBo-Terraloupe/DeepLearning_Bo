#
# Schema definitions for meta data
#

base_dataset_schema = {
    "name": {
        "type": str,
        "required": True,
        "description": "Dataset name"
    },
    "data": {
        "type": {
            "description": "Type of the images (e.g. nadir, ortho, true-ortho, oblique)",
            "enum": ["nadir", "true-ortho", "ortho", "oblique"],
            "required": True
        },
        "channels": {
            "type": str,
            "description": "Channels, that are present in the image (e.g. rgb, rgbi)",
            "default": "rgb"
        },
        "source_bitdepth": {
            "type": int,
            "description": "Bits per channel",
            "default": 8
        },
    },
    "ground_truth": {
        "type": {
            "description": "Type of the ground thruth data (e.g. semseg for semantic segmentation)",
            "enum": ["semseg", "detection"],
            "default": "semseg"
        },
        "classes": {
            "type": list,
            "description": "Ground truth classes, list of class name and index",
        }
    }

}

image_dataset_schema = base_dataset_schema.copy()
image_dataset_schema.update({
    "training": {
        "images": {
            "type": list,
            "description": "List of image files, to be used for training. Relative to current directory",
        },
        "labels": {
            "type": list,
            "description": "List of label files, to be used for training. Relative to current directory",
        }
    },
    "validation": {
        "images": {
            "type": list,
            "description": "List of image files, to be used for validation. Can be empty, of train/validation split is performed externally. Relative to current directory",
        },
        "labels": {
            "type": list,
            "description": "List of label files, to be used for testing. Relative to current directory",
        }
    },
    "test": {
        "images": {
            "type": list,
            "description": "List of image files, to be used for testing. Relative to current directory",
        },
        "labels": {
            "type": list,
            "description": "List of label files, to be used for testing. Relative to current directory",
        }
    }
})

patch_dataset_schema = base_dataset_schema.copy()
patch_dataset_schema.update({
    "training": {
        "blocksize": {
            "type": int,
            "description": "Blocksize or patchsize. Number of pixels of a square patch",
            "default": 64,
        },
        "file": {
            "type": file,
            "exists": True,
            "description": "Datafile in numpy format",
            "required": True,
        },
        "flip_channels": {
            "type": bool,
            "description": "Inverse channel order. Userful for bgr trained networks.",
            "default": False
        },
        "mean_color": {
            "type": list,
            "description": "Mean value for each channel, list of floats between [0,1]",
            "default": False
        },
        "ratio": {
            "type": float,
            "description": "Negative ratio",
            "default": False
        },
        "granularity_level": {
            "description": "Negative ratio",
            "default": 'patch',
            "enum": ["patch", "pixel"]
        },
        "images": {
            "type": list,
            "description": "List of image names, from which the training set was generated.",
        },
    }
})
