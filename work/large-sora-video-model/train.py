import argparse
import mlflow

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--prompt', default='a cinematic product demo')
    parser.add_argument('--duration-seconds', type=int, default=4)
    parser.add_argument('--model-path', default='model/sora-video-sample.onnx')
    args = parser.parse_args()
    with mlflow.start_run():
        mlflow.log_param('model_family', 'sora-style-video-generation')
        mlflow.log_param('prompt', args.prompt)
        mlflow.log_param('duration_seconds', args.duration_seconds)
        mlflow.log_artifact(args.model_path)

if __name__ == '__main__':
    main()
