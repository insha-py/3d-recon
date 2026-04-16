# 3D Reconstruction using VGGT

Pipeline:

Images -> VGGT (Kaggle GPU) -> pose + depth outputs -> local parsing -> point cloud -> visualization

Structure:

src/
    vggt/          # COLMAP parsing
    datasets/      # dataset-specific loaders (ADVIO)
    geometry/      # point cloud + pose handling
    imu/           # IMU pose priors + keyframe selection
    visualization/ # visualization tools

data/scans/       # raw and reconstructed scans
outputs/          # exported formats

## Local sparse reconstruction parsing

```bash
python process_scan.py --scan scan_001
```

## IMU-guided VGGT workflow

VGGT can be compared against a `pose/` folder, but it does not natively consume IMU priors inside the transformer. In this repo, IMU helps in two practical ways:

1. Build per-frame camera pose priors from IMU timestamps and orientation.
2. Reduce GPU usage by selecting fewer, motion-informative frames before inference.

### Expected metadata

`imu.csv`

- Must include a timestamp column.
- Must include either quaternion columns (`qw,qx,qy,qz`) or gyro columns (`gyro_x,gyro_y,gyro_z`).
- Optional position columns (`pos_x,pos_y,pos_z`) improve translation priors.

`frame_times.csv`

- Must include `image` or `image_name`.
- Must include a timestamp column aligned to the same clock as the IMU.

### Build an IMU-guided dataset

```bash
python prepare_imu_dataset.py ^
  --images-dir data/my_scene/images ^
  --imu-csv data/my_scene/imu.csv ^
  --frame-times-csv data/my_scene/frame_times.csv ^
  --output-root data/my_scene_vggt_imu
```

This creates a VGGT-ready reduced dataset:

- `images/`: selected keyframes only
- `pose/`: IMU-derived camera-to-world priors as `*.txt`
- `metadata/selected_frames.csv`: mapping from original images to selected frames

### Compare baseline vs IMU-guided runs

After running VGGT once on the full image set and once on the IMU-guided dataset:

```bash
python compare_pose_runs.py ^
  --baseline-dir outputs/baseline/poses ^
  --imu-guided-dir outputs/imu_guided/poses ^
  --prior-dir data/my_scene_vggt_imu/pose
```

### Recommended experiment

1. Run baseline VGGT on all frames.
2. Run `prepare_imu_dataset.py` to create the reduced IMU-guided dataset.
3. Run VGGT on the reduced dataset.
4. Compare pose drift against the IMU priors with `compare_pose_runs.py`.
5. Track GPU memory and runtime in Kaggle for both runs.

### Important limitation

IMU priors alone do not reduce VGGT compute unless you either:

- reduce the number of frames before inference, or
- modify VGGT itself to inject priors into the camera head or token selection logic.

The current implementation supports the first path cleanly and gives you a fair baseline-vs-IMU-guided comparison.

## ADVIO benchmark workflow

ADVIO is a better fit than ad hoc phone JPEGs when you want to study how IMU helps or does not help a VGGT pipeline, because it includes:

- synchronized camera frame timestamps
- accelerometer and gyroscope streams
- ground-truth poses
- sequence-level tags such as stairs, escalators, elevators, and crowd level

Expected ADVIO folder layout:

```text
data/advio-01/
  ground-truth/
    poses.csv
  iphone/
    frames.mov
    frames.csv
    accelerometer.csv
    gyroscope.csv
    arkit.csv        # optional
```

### Prepare one ADVIO sequence for VGGT

This extracts the iPhone video into images, keeps a baseline copy of all extracted frames, and builds an IMU-guided reduced subset plus per-frame pose priors.

```bash
python prepare_advio_vggt_dataset.py ^
  --advio-root data/advio-01 ^
  --output-root data/prepared/advio-01 ^
  --every-nth-frame 2 ^
  --rotation-thresh-deg 4.0 ^
  --min-gap 1
```

Outputs:

- `baseline/images/`
- `baseline/metadata/frame_times.csv`
- `imu_guided/images/`
- `imu_guided/pose/`
- `imu_guided/metadata/selected_frames.csv`
- `summary.json`

### Run VGGT

1. Run VGGT on `baseline/images/`
2. Run VGGT on `imu_guided/images/`
3. Save the predicted poses from each run as text matrices

The benchmark script accepts either:

- `*_extrinsic.txt` files from a VGGT-style export, or
- plain `*.txt` camera-to-world matrices

### Benchmark baseline vs IMU-guided runs

```bash
python benchmark_advio_vggt.py ^
  --advio-root data/advio-01 ^
  --baseline-frames-csv data/prepared/advio-01/baseline/metadata/frame_times.csv ^
  --baseline-pose-dir outputs/advio-01/baseline/poses ^
  --imu-guided-frames-csv data/prepared/advio-01/imu_guided/metadata/selected_frames.csv ^
  --imu-guided-pose-dir outputs/advio-01/imu_guided/poses ^
  --output-json outputs/analysis/advio-01_benchmark.json
```

The report includes:

- absolute trajectory error after similarity alignment
- relative rotation and translation error
- turn-segment metrics using gyro magnitude
- vertical-motion metrics as a stairs/elevator proxy
- sequence tags from ADVIO metadata
- GPU-use proxies from frame count reduction

### GPU-use interpretation

The benchmark reports two simple GPU proxies:

- `gpu_frame_proxy_ratio`: selected frame count divided by baseline frame count
- `gpu_attention_proxy_ratio`: squared frame-count ratio

These are not direct measurements from a profiler. They are proxies for how much sequence reduction might reduce memory and compute in a transformer-style vision pipeline.

### Notes

- ADVIO gives a clean way to measure whether IMU helps pose quality, robustness, or compute on real phone data.
- VGGT still remains vision-only unless you modify the model itself; the current setup uses IMU for priors, frame selection, and evaluation.
- Sequence tags such as stairs, elevators, and crowd level come from the ADVIO benchmark metadata and help interpret failures and gains by scenario.
