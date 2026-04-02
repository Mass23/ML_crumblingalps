# old version
#python3 download_images.py
#python3 create_landslide_traindata.py --input-dir data/raw-images --output-dir data/training --fps 20 --target-width 2880 --target-height 1620 --num-workers 3 --duration 10


# Process 5 images → generate data → train → checkpoint → repeat
python train.py --mode batch \
  --images-dir data/raw-images \
  --data-dir data/training \
  --batch-size 5 \
  --num-epochs 10

python train.py --mode batch \
  --images-dir data/raw-images \
  --data-dir data/training \
  --batch-size 5 \
  --num-epochs 10

python train.py --mode batch \
  --images-dir data/raw-images \
  --data-dir data/training \
  --batch-size 5 \
  --num-epochs 10

python train.py --mode batch \
  --images-dir data/raw-images \
  --data-dir data/training \
  --batch-size 5 \
  --num-epochs 10
