Запускать надлежит следующим образом:

`python3 -m fine-tuning-experiments.train`

Необходим правильный проепроцессинг. Можно скачать CSN и положить датасет выбранного языка в `data/raw/csn`, после чего выполнить скрипт:

`./fine-tuning-experiments/preprocess_python.sh`

Временные указания насчёт костылей:
* Следить за `TODO` в репозитории (особенно при смене датасетов)
* Возвращение к _method name prediction_ осуществляется прежде всего указанием в `code_transformer.preprocessing.pipeline.stage1.CTStage1Sample.__init__` по умолчанию `use_docstrings=False`, исключением из `code_transformer.preprocessing.pipeline.stage2.CTStage2Sample` строки `self.func_name = docstring` и заменой в `code_transformer.preprocessing.datamanager.csn.raw.CSNRawDataLoader.read` строчки `reader = map(lambda line:...` на стандартный вариант оной