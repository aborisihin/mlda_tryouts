AUTOML_SETTINGS_FILE=$(CURDIR)/settings/automl.json
DATA_PREPARE_SETTINGS_FILE=$(CURDIR)/settings/data_prepare.json

define get_setting
$(shell python ./src/utils/settings_reader.py $(AUTOML_SETTINGS_FILE) $(1))
endef

# params
MODEL_NAME=$(call get_setting, model/name)
DOCKER_IMAGE=$(call get_setting, docker/image)
SOLVER=$(call get_setting, solver/type)
TIME_LIMIT_SEC=$(call get_setting, solver/time_limit_sec)
MEMORY_LIMIT_MB=$(call get_setting, solver/memory_limit_mb)
CLASSIFICATION_METRICS = $(call get_setting, solver/metrics/classification)
CLASSIFICATION_NEED_PROBA = $(call get_setting, solver/metrics/classification_need_proba)
REGRESSION_METRICS = $(call get_setting, solver/metrics/regression)
DATA_PATH=$(call get_setting, data/path)
TRAIN_FILE=$(call get_setting, data/train_file)
TEST_FILE=$(call get_setting, data/test_file)
VALIDATION_FILE=$(call get_setting, data/validation_file)
SAVE_PROCESSED_DATA=$(call get_setting, data/save_processed_data)

# docker static commands
DOCKER_BUILD=docker build -t ${DOCKER_IMAGE} . && (docker ps -q -f status=exited | xargs docker rm) && (docker images -qf dangling=true | xargs docker rmi) && docker images
DOCKER_RUN=docker run --rm -it -v ${CURDIR}:/app -v ${DATA_PATH}:/app/data -w /app

# static paths
TRAIN_CSV=/app/data/${TRAIN_FILE}
TEST_CSV=/app/data/${TEST_FILE}
PREDICTIONS_CSV=/app/predictions/prediction_$(shell /bin/date "+%Y%m%d_%H%M%S").csv
MODEL_DIR=models/model_${MODEL_NAME}

ifneq ($(strip $(VALIDATION_FILE)), None)
	VALIDATION_CSV=/app/data/${VALIDATION_FILE}
else
	VALIDATION_CSV=None
endif

# makefile commands
model-fit-classification:
	${DOCKER_RUN} ${DOCKER_IMAGE} python3 ./src/fit.py --solver $(SOLVER) --mode classification --metrics $(CLASSIFICATION_METRICS) --train-csv ${TRAIN_CSV} --model-dir ${MODEL_DIR} --time-limit ${TIME_LIMIT_SEC} --memory-limit ${MEMORY_LIMIT_MB} --save-processed-data ${SAVE_PROCESSED_DATA}

model-fit-regression:
	${DOCKER_RUN} ${DOCKER_IMAGE} python3 ./src/fit.py --solver $(SOLVER) --mode regression --metrics $(REGRESSION_METRICS) --train-csv ${TRAIN_CSV} --model-dir ${MODEL_DIR} --time-limit ${TIME_LIMIT_SEC} --memory-limit ${MEMORY_LIMIT_MB} --save-processed-data ${SAVE_PROCESSED_DATA}

model-predict:
	${DOCKER_RUN} ${DOCKER_IMAGE} python3 ./src/predict.py --solver $(SOLVER) --test-csv ${TEST_CSV} --prediction-csv ${PREDICTIONS_CSV} --validation-csv ${VALIDATION_CSV} --model-dir ${MODEL_DIR} --need-proba $(CLASSIFICATION_NEED_PROBA)

data-prepare:
	${DOCKER_RUN} ${DOCKER_IMAGE} python3 ./src/data_prepare.py --settings $(DATA_PREPARE_SETTINGS_FILE)

docker-build:
	${DOCKER_BUILD}

docker-push:
	docker push ${DOCKER_IMAGE}

run-bash:
	${DOCKER_RUN} ${DOCKER_IMAGE} /bin/bash

run-jupyter:
	${DOCKER_RUN} -p 8889:8889 ${DOCKER_IMAGE} jupyter notebook --ip=0.0.0.0 --port=8889 --no-browser --allow-root  --NotebookApp.token='' --NotebookApp.password=''
