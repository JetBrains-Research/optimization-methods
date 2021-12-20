from code_transformer.experiments.experiment_ddp import ExperimentSetupDDP, ex
from code_transformer.experiments.mixins.code_summarization import CTCodeSummarizationMixin
from code_transformer.experiments.mixins.code_trans_transformer import CodeTransformerDecoderMixin


class CodeTransDecoderExperimentSetup(CodeTransformerDecoderMixin,
                                      CTCodeSummarizationMixin,
                                      ExperimentSetupDDP):
    pass


@ex.automain
def main():
    experiment = CodeTransDecoderExperimentSetup()
    experiment.train()


@ex.command(unobserved=True)
def recreate_experiment():
    return CodeTransDecoderExperimentSetup()