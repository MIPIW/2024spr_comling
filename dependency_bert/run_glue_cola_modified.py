
import logging
import os
import random
import sys
from dataclasses import dataclass, field
from typing import Optional

import datasets
import evaluate
import numpy as np
from datasets import load_dataset

import transformers
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EvalPrediction,
    HfArgumentParser,
    PretrainedConfig,
    Trainer,
    TrainingArguments,
    default_data_collator,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version, send_example_telemetry
from transformers.utils.versions import require_version

task_to_keys = {
  "cola": ("sentence", None),
  "mnli": ("premise", "hypothesis"),
  "mrpc": ("sentence1", "sentence2"),
  "qnli": ("question", "sentence"),
  "qqp": ("question1", "question2"),
  "rte": ("sentence1", "sentence2"),
  "sst2": ("sentence", None),
  "stsb": ("sentence1", "sentence2"),
  "wnli": ("sentence1", "sentence2"),
}

logger = logging.getLogger(__name__)

@dataclass
class DataTrainingArguments:
  """
  Arguments pertaining to what data we are going to input our model for training and eval.

  Using `HfArgumentParser` we can turn this class
  into argparse arguments to be able to specify them on
  the command line.
  """

  task_name: Optional[str] = field(
      default=None,
      metadata={"help": "The name of the task to train on: " + ", ".join(task_to_keys.keys())},
  )
  dataset_name: Optional[str] = field(
      default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
  )
  dataset_config_name: Optional[str] = field(
      default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
  )
  max_seq_length: int = field(
      default=128,
      metadata={
          "help": (
              "The maximum total input sequence length after tokenization. Sequences longer "
              "than this will be truncated, sequences shorter will be padded."
          )
      },
  )
  overwrite_cache: bool = field(
      default=False, metadata={"help": "Overwrite the cached preprocessed datasets or not."}
  )
  pad_to_max_length: bool = field(
      default=True,
      metadata={
          "help": (
              "Whether to pad all samples to `max_seq_length`. "
              "If False, will pad the samples dynamically when batching to the maximum length in the batch."
          )
      },
  )
  max_train_samples: Optional[int] = field(
      default=None,
      metadata={
          "help": (
              "For debugging purposes or quicker training, truncate the number of training examples to this "
              "value if set."
          )
      },
  )
  max_eval_samples: Optional[int] = field(
      default=None,
      metadata={
          "help": (
              "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
              "value if set."
          )
      },
  )
  max_predict_samples: Optional[int] = field(
      default=None,
      metadata={
          "help": (
              "For debugging purposes or quicker training, truncate the number of prediction examples to this "
              "value if set."
          )
      },
  )
  train_file: Optional[str] = field(
      default=None, metadata={"help": "A csv or a json file containing the training data."}
  )
  validation_file: Optional[str] = field(
      default=None, metadata={"help": "A csv or a json file containing the validation data."}
  )
  test_file: Optional[str] = field(default=None, metadata={"help": "A csv or a json file containing the test data."})

  def __post_init__(self):
      if self.task_name is not None:
          self.task_name = self.task_name.lower()
          if self.task_name not in task_to_keys.keys():
              raise ValueError("Unknown task, you should pick one in " + ",".join(task_to_keys.keys()))
      elif self.dataset_name is not None:
          pass
      elif self.train_file is None or self.validation_file is None:
          raise ValueError("Need either a GLUE task, a training/validation file or a dataset name.")
      else:
          train_extension = self.train_file.split(".")[-1]
          assert train_extension in ["csv", "json"], "`train_file` should be a csv or a json file."
          validation_extension = self.validation_file.split(".")[-1]
          assert (
              validation_extension == train_extension
          ), "`validation_file` should have the same extension (csv or json) as `train_file`."



@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    token: str = field(
        default=None,
        metadata={
            "help": (
                "The token to use as HTTP bearer authorization for remote files. If not specified, will use the token "
                "generated when running `huggingface-cli login` (stored in `~/.huggingface`)."
            )
        },
    )
    trust_remote_code: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether or not to allow for custom models defined on the Hub in their own modeling files. This option "
                "should only be set to `True` for repositories you trust and in which you have read the code, as it will "
                "execute code present on the Hub on your local machine."
            )
        },
    )
    ignore_mismatched_sizes: bool = field(
        default=False,
        metadata={"help": "Will enable to load a pretrained model whose head dimensions are different."},
    )

def main():
  parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
  model_args, data_args, training_args = parser.parse_args_into_dataclasses()

  send_example_telemetry("run_glue", model_args, data_args)

  #로깅 셋업
  logging.basicConfig(
      format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
      datefmt="%m/%d/%Y %H:%M:%S",
      handlers=[logging.StreamHandler(sys.stdout)],
  )
  transformers.utils.logging.set_verbosity_info()
  log_level = training_args.get_process_log_level()
  logger.setLevel(log_level)
  datasets.utils.logging.set_verbosity(log_level)
  transformers.utils.logging.set_verbosity(log_level)
  transformers.utils.logging.enable_default_handler()
  transformers.utils.logging.enable_explicit_format()

  # Log on each process the small summary:
  logger.warning(
      f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}, "
      + f"distributed training: {training_args.parallel_mode.value == 'distributed'}, 16-bits training: {training_args.fp16}"
  )
  logger.info(f"Training/evaluation parameters {training_args}")

  #체크포인트
  last_ckpt = None
  if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
    last_ckpt = get_last_checkpoint(training_args.output_dir)
    if last_ckpt is None and len(os.listdir(training_args.output_dir)) > 0:
      raise ValueError(
          f'output directory ({training_args.output_dir}) already exists and is not empty'
          'use --overwrite_output_dir to overcome'
      )
    elif last_ckpt is not None and training_args.resume_from_checkpoint is None:
      logger.info(
          f'checkpoint detected, resuming training at {last_ckpt}. To avoid this behavior, change'
          'the --output_dir or add --overwrite_output_dir to train from scratch'
      )

  set_seed(training_args.seed)

  # 데이터셋 로드
  if data_args.task_name is not None:
    raw_datasets = load_dataset('nyu-mll/glue', data_args.task_name, cache_dir = model_args.cache_dir, token = model_args.token)
  else:
    data_files = {'train': data_args.train_file, 'validation': data_args.validation_file}
    if training_args.do_predict:
      if data_args.test_file is not None:
        train_extension = data_args.train_file.split('.')[-1]
        test_extension = data_args.test_file.split('.')[-1]
        assert (test_extension == train_extension), "test_file should have the same extension as train_file"
      else: raise ValueError("Need either GLUE task or a test file for do_predict")

    for key in data_files.keys():
      logger.info(f'load a local file for {key}: {data_files[key]}')

    if data_args.train_file.endswith('.csv'):
      raw_datasets = load_dataset('csv', data_files = data_files, cache_dir = model_args.cache_dir, token = model_args.token)
    else:
      raw_datasets = load_dataset('json', data_files = data_files, cache_dir = model_args.cache_dir, token = model_args.token)


   # 레이블 처리 (태스크별)
  if data_args.task_name is not None:
    is_regression = data_args.task_name == 'stsb' #STS-B 셋만 회귀, 나머지는 분류 태스크
    if not is_regression: #분류 태스크
      label_list = raw_datasets['train'].features['label'].names
      num_labels = len(label_list)
    else:
      num_labels = 1

  else: #직접 파일 넣은 경우
    is_regression = raw_datasets['train'].features['label'].dtype in ['float32','float64']
    if is_regression: num_labels = 1
    else:
      label_list = raw_datasets['train'].unique('label')
      label_list.sort()
      num_labels = len(label_list)


  # 모델 로드
  config = AutoConfig.from_pretrained(
      model_args.config_name if model_args.config_name else model_args.model_name_or_path,
      num_labels = num_labels,
      finetuning_task = data_args.task_name,
      cache_dir = model_args.cache_dir,
      revision = model_args.model_revision,
      token = model_args.token,
      trust_remote_code = model_args.trust_remote_code,
  )

  tokenizer = AutoTokenizer.from_pretrained(
      model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
      cache_dir = model_args.cache_dir,
      use_fast = model_args.use_fast_tokenizer,
      revision = model_args.model_revision,
      token = model_args.token,
      trust_remote_code = model_args.trust_remote_code,
  )

  model = AutoModelForSequenceClassification.from_pretrained(
      model_args.model_name_or_path,
      from_tf = bool('.ckpt' in model_args.model_name_or_path),
      config = config,
      cache_dir = model_args.cache_dir,
      revision = model_args.model_revision,
      token = model_args.token,
      trust_remote_code = model_args.trust_remote_code,
      ignore_mismatched_sizes = model_args.ignore_mismatched_sizes,
      use_safetensors = bool(model_args.model_name_or_path != 'bert-base-uncased'), #safetensor로 된 로컬 체크포인트만 해당
  )

  #전처리
  if data_args.task_name is not None:
    sentence1_key, sentence2_key = task_to_keys[data_args.task_name]
  else:
    non_label_column_names = [name for name in raw_datasets['train'].column_names if name != 'label']
    sentence1_key = non_label_column_names[0]
    if len(non_label_column_names) < 2:
      sentence2_key = None
    else:
      sentence2_key = non_label_column_names[1]

  #패딩
  if data_args.pad_to_max_length: padding = 'max_length'
  else: padding = False

  # 레이블 정리
  label_to_id = None
  if (model.config.label2id != PretrainedConfig(num_labels = num_labels).label2id and data_args.task_name is not None and not is_regression):
    label_name_to_id = {k.lower(): v for k, v in model_args.config.label2id.items()}
    if sorted(label_name_to_id.keys()) == sorted(label_list):
      label_to_id = {i: int(label_name_to_id[label_list[i]]) for i in range(num_labels)}
    else:
      logger.warning('your model seems to have been trained with labels, but they don\'t match the dataset:',
                     f'model labels: {sorted(label_name_to_id.keys())}, dataset labels: {sorted(label_list)}',
                     'Ignoring the model labels as a result.',
                     )
  elif data_args.task_name is None and not is_regression: #태스크 네임 없고 분류 태스크인 경우
    label_to_id = {v: i for i, v in enumerate(label_list)}

  if label_to_id is not None:
    model.config.label2id = label_to_id
    model.config.id2label = {id: label for label, id in config.label2id.items()}
  elif data_args.task_name is not None and not is_regression: # 태스크 네임 있고 분류 태스크인 경우
    model.config.label2id = {l: i for i, l in enumerate(label_list)}
    model.config.id2label = {id: label for label, id in config.label2id.items()}

  if data_args.max_seq_length > tokenizer.model_max_length:
    logger.warning(
        f'the max_seq_length passed is larger than the max length for the model. Using max_seq_length = {tokenizer.model_max_length}.'
    )
  max_seq_length = min(data_args.max_seq_length, tokenizer.model_max_length)


  ### 전처리 함수
  def preprocess_function(examples):
    args = (
        (examples[sentence1_key],) if sentence2_key is None else (examples[sentence1_key], examples[sentence2_key])
    )
    result = tokenizer(*args, padding = padding, max_length = max_seq_length, truncation = True)
    return result

  # 데이터 전처리 적용
  with training_args.main_process_first(desc = 'dataset map pre-processing'):
    raw_datasets = raw_datasets.map(preprocess_function, batched = True, load_from_cache_file = not data_args.overwrite_cache)

  # train / dev / test set 와꾸맞추기..
  if training_args.do_train:
    if 'train' not in raw_datasets:
      raise ValueError('--do_train requires a train dataset')
    train_dataset = raw_datasets['train']
    if data_args.max_train_samples is not None:
      max_train_samples = min(len(train_dataset), data_args.max_train_samples)
      train_dataset = train_dataset.select(range(max_train_samples))

  if training_args.do_eval:
    if 'validation' not in raw_datasets and 'validation_matched' not in raw_datasets:
      raise ValueError('--do_eval requires a validation dataset')
    eval_dataset = raw_datasets['validation_matched' if data_args.task_name == 'mnli' else 'validation']
    if data_args.max_eval_samples is not None:
      max_eval_samples = min(len(eval_dataset), data_args.max_eval_samples)
      eval_dataset = eval_dataset.select(range(max_eval_samples))

  if training_args.do_predict or data_args.task_name is not None or data_args.test_file is not None:
    if 'test' not in raw_datasets and 'test_matched' not in raw_datasets:
      raise ValueError('--do_predict requires a test dataset')
    predict_dataset = raw_datasets['test_matched' if data_args.task_name == 'mnli' else 'test']
    if data_args.max_predict_samples is not None:
      max_predict_samples = min(len(predict_dataset), data_args.max_predict_samples)
      predict_dataset = predict_dataset.select(range(max_predict_samples))


  # 평가 메트릭 정의
  if data_args.task_name is not None:
    metric = evaluate.load('glue', data_args.task_name, cache_dir = model_args.cache_dir)
    if data_args.task_name == 'cola':
        metric = evaluate.load('accuracy', cache_dir = model_args.cache_dir) 	
  elif is_regression: #회귀 평가메트릭: mse
    metric = evaluate.load('mse', cache_dir = model_args.cache_dir)
  else:
    metric = evaluate.load('accuracy', cache_dir = model_args.cache_dir)

  def compute_metrics(p: EvalPrediction):
    preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
    preds = np.squeeze(preds) if is_regression else np.argmax(preds, axis = 1)
    result = metric.compute(predictions = preds, references = p.label_ids)
    if len(result) > 1:
      result['combined_score'] = np.mean(list(result.values())).item()
    return result


  if data_args.pad_to_max_length: data_collator = default_data_collator
  elif training_args.fp16: data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of = 8)
  else: data_collator = None


  # 트레이너 initialize
  trainer = Trainer(
      model = model,
      args = training_args,
      train_dataset = train_dataset if training_args.do_train else None,
      eval_dataset = eval_dataset if training_args.do_eval else None,
      compute_metrics = compute_metrics,
      tokenizer = tokenizer,
      data_collator = data_collator,
  )


  # 학습
  if training_args.do_train:
    checkpoint = None
    if training_args.resume_from_checkpoint is not None:
      checkpoint = training_args.resume_from_checkpoint
    elif last_ckpt is not None:
      checkpoint = last_ckpt
    train_result = trainer.train(resume_from_checkpoint = checkpoint)
    metrics = train_result.metrics
    max_train_samples = (
        data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
    )
    metrics['train_samples'] = min(max_train_samples, len(train_dataset))

    trainer.save_model()
    trainer.log_metrics('train', metrics)
    trainer.save_metrics('train', metrics)
    trainer.save_state()


  # eval
  if training_args.do_eval:
    logger.info("*** Evaluate ***")
    tasks = [data_args.task_name]
    eval_datasets = [eval_dataset]
    if data_args.task_name == 'mnli':
      tasks.append('mnli-mm')
      valid_mm_dataset = raw_datasets['validation_mismatched']
      if data_args.max_eval_samples is not None:
        max_eval_samples = min(len(valid_mm_dataset), data_args.max_eval_samples)
        valid_mm_dataset = valid_mm_dataset.select(range(max_eval_samples))
      eval_datasets.append(valid_mm_dataset)
      combined = {}

    for eval_dataset, task in zip(eval_datasets, tasks):
      metrics = trainer.evaluate(eval_dataset = eval_dataset)
      max_eval_samples = (data_args.max_eval_samples if data_args.max_eval_samples is not None else len(eval_dataset))
      metrics['eval_samples'] = min(max_eval_samples, len(eval_dataset))

      if task == 'mnli-mm':
        metrics = {k + '_mm': v for k, v in metrics.items()}
      if task is not None and 'mnli' in task:
        combined.update(metrics)

      trainer.log_metrics('eval', metrics)
      trainer.save_metrics('eval', combined if task is not None and 'mnli' in task else metrics)


  # predict
  if training_args.do_predict:
    logger.info("*** Predict ***")
    tasks = [data_args.task_name]
    predict_datasets = [predict_dataset]
    if data_args.task_name == 'mnli':
      tasks.append('mnli-mm')
      predict_datasets.append(raw_datasets['test_mismatched'])

    for predict_dataset, task in zip(predict_datasets, tasks):
      predict_dataset = predict_dataset.remove_columns('label')
      predictions = trainer.predict(predict_dataset, metric_key_prefix = 'predict').predictions
      predictions = np.squeeze(predictions) if is_regression else np.argmax(predictions, axis=1)
      output_predict_file = os.path.join(training_args.output_dir, f'predict_results_{task}.txt')
      if trainer.is_world_process_zero():
        with open(output_predict_file, 'w') as writer:
          logger.info(f'**** Predict results {task} *****')
          writer.write('index\tprediction\n')
          for index, item in enumerate(predictions):
            if is_regression:
              writer.write(f'{index}\t{item:3.3f}\n')
            else:
              item = label_list[item]
              writer.write(f'{index}\t{item}\n')


  # 모델 카드 생성
  kwargs = {'finetuned_from': model_args.model_name_or_path, 'tasks': 'text-classification'}
  if data_args.task_name is not None:
    kwargs['language'] = 'en'
    kwargs['dataset_tags'] = 'glue'
    kwargs['dataset_args'] = data_args.task_name
    kwargs['dataset'] = f'GLUE {data_args.task_name.upper()}'

  if training_args.push_to_hub:
    trainer.push_to_hub(**kwargs)
  else:
    trainer.create_model_card(**kwargs)

if __name__ == '__main__':
  main()

