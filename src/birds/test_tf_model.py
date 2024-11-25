import sys

from datetime import datetime as dt
from pathlib import Path

from birds.lib.image_detection import (ExperimentSummaryStats, detect)
from birds.lib.logger import get_logger, update_app_verbosity_level
from birds.lib.tf_models import load_model


logger = get_logger("test_tf_pipeline", verbosity=2)
update_app_verbosity_level(verbosity=2)


def main(input_dir: Path):
    input_images = list(input_dir.glob("*.jpg")) + list(input_dir.glob("*.png"))
    # TODO: sort `input_images`; keep append-only file for all processed imanges
    num_input_images = len(input_images)
    logger.info(f"Testing {num_input_images} images")

    experiment_id = f"{input_dir.name}_experiment-{dt.now().strftime('%Y-%m-%dT%H-%M')}"

    all_summary_stats = []

    for model_version in [
        # "d4",  # "d4" is buggy (zero detection rate)
        # "d5",
        "d6",
        # "d7",
    ]:
        model = load_model(family="efficientdet", version=model_version)
        logger.info(f"Now testing images with 'efficientdet/{model_version}'")

        model_results = []

        for job_index, image_path in enumerate(input_images):
            # Log progress at ~ every 10% of the input dataset processed.
            if job_index % max(num_input_images // 10, 1) == 0:
                logger.info(f"Now processing image '{image_path}'")

            image_file = image_path.resolve()
            result = detect(model, image_file)
            model_results.append(result)

        model_output_dir = Path(f"test-detection/model-outputs/{experiment_id}/efficientdet-{model_version}")
        if not model_output_dir.exists():
            model_output_dir.mkdir(parents=True)

        logger.info(f"{model_version}: Saving labeled images to the file system")
        for r in model_results:
            r.save_categorized_results(base_dir=str(model_output_dir))

        model_summary_stats = ExperimentSummaryStats.summarize_detection_results(
            model_version=model_version,
            results=model_results)
        all_summary_stats.append(model_summary_stats)

        print(model_summary_stats)
        print("\n")

    ExperimentSummaryStats.display_stats_table(all_summary_stats)


if __name__ == "__main__":
    input_dir = Path(sys.argv[1])
    if not input_dir.exists() or not input_dir.is_dir():
        raise RuntimeError(f"No such dir exists: '{input_dir}'")

    main(input_dir)
