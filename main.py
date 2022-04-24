import hydra
from time import time
from datetime import datetime
import os
import logging

from src.data.audio import Audio
from src.data.onset import Onset
from src.transformer.audio_tf import AudioTransformer
from src.transformer.onset_tf import OnsetTransformer
from src.model import Model
from src.trainer import Trainer
from src import utils


log = logging.getLogger(__name__)


@hydra.main(config_path="configs/Piano.yml")
def main(conf):
    log.info("Start program")
    time_start = time()

    # load audio and onset list file
    original_work_dir = hydra.utils.get_original_cwd()
    audio_list_path = "/".join([original_work_dir, conf.input.audio])
    onset_list_path = "/".join([original_work_dir, conf.input.onset])
    audio_path_list = utils.load_list_file(audio_list_path)
    onset_path_list = utils.load_list_file(onset_list_path)

    for audio_path, onset_path in zip(audio_path_list, onset_path_list):
        audio_name = os.path.splitext(os.path.basename(audio_path))[0]
        # Load audio
        #audio_path = "/".join([original_work_dir, audio_path])
        audio_path = audio_path
        audio = Audio.load(audio_path, **conf.audio)
        # Load onset
        #onset_path = "/".join([original_work_dir, onset_path])
        onset_path = onset_path
        onset = Onset.load(onset_path, conf.audio.duration)

        # Transform audio
        audio_transformer = AudioTransformer(**conf.transform_audio)
        X_mag, X_phase = audio_transformer.transform(audio.y)

        # Transform onset
        onset_transformer = OnsetTransformer(conf.model_params.K, conf.transform_audio.stft_hop_length, **conf.transform_onset)

        # Repeat inference of OI-NMF model (experiment_num) times.
        trainer = Trainer()
        trainer.set_config(conf)
        for i in range(conf.others.experiment_num):
            log.info("--- Process: %s/%s ---" % (i+1, conf.others.experiment_num))
            onset_matrix = onset_transformer.transform(onset, audio)
            model = Model(X_mag, X_phase, onset_matrix)
            trainer.set_model(model)
            trainer.train(conf.train.method)
            model.save("{}_{}.npz".format(audio_name, i))

    log.info("Finish program")
    log.info("Elapsed time: %.3f [sec]" % (time() - time_start))


if __name__ == "__main__":
    main()
