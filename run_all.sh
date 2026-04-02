# old version
#python3 download_images.py
#python3 create_landslide_traindata.py --input-dir data/raw-images --output-dir data/training --fps 20 --target-width 2880 --target-height 1620 --num-workers 3 --duration 10


# Process 5 images → generate data → train → checkpoint → repeat
python3 train.py --images-dir data/raw-images --data-dir data/training --batch-size 5 --num-epochs 20
