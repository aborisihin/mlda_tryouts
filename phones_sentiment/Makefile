SETTINGS_FILE=$(CURDIR)/settings/sentiment.json

define get_setting
$(shell python ./src/utils/settings_reader.py $(SETTINGS_FILE) $(1))
endef

# params
DOCKER_IMAGE=$(call get_setting, docker/image)
SCRAPPING_START_PAGE=$(call get_setting, scrapping/start_page)
SCRAPPING_DOWNLOAD_DELAY=$(call get_setting, scrapping/download_delay)
HYPEROPT_MAX_EVALS=$(call get_setting, model_fitting/hyperopt_max_evals)

# docker static commands
DOCKER_BUILD=docker build -t ${DOCKER_IMAGE} . && (docker ps -q -f status=exited | xargs docker rm) && (docker images -qf dangling=true | xargs docker rmi) && docker images
DOCKER_RUN=docker run --rm -it -v ${CURDIR}:/app

# makefile commands
scrapy-run:
	${DOCKER_RUN} -w /app/src/scrapping ${DOCKER_IMAGE} scrapy crawl yandex_mobile -a start_page=${SCRAPPING_START_PAGE} -s DOWNLOAD_DELAY=${SCRAPPING_DOWNLOAD_DELAY}

model-fit:
	${DOCKER_RUN} -w /app ${DOCKER_IMAGE} python3 ./src/fit.py --hyperopt_max_evals $(HYPEROPT_MAX_EVALS)

ui-run:
	${DOCKER_RUN} -w /app ${DOCKER_IMAGE} python3 ./src/ui/flask_ui.py

docker-build:
	${DOCKER_BUILD}

docker-push:
	docker push ${DOCKER_IMAGE}

docker-bash:
	${DOCKER_RUN} -w /app ${DOCKER_IMAGE} /bin/bash
