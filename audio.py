import librosa
import numpy as np
import pickle
from librosa.feature.rhythm import tempo as librosa_tempo


class AudioFeature:
    def __init__(self,
                 path,
                 genre,
                 duration=10,
                 offset=25):
        """
        Keep duration num seconds of each clip, starting at
        offset num seconds into the song (avoid intros)
        """
        self.path = path
        self.genre = genre
        self.y, self.sr = librosa.load(self.path,
                                       duration=duration,
                                       offset=offset)
        self.features = None

    def _concat_features(self, feature):
        """
        Whenever an _extract_X method is called by main.py,
        this helper function concatenates to Audio instance
        features attribute
        """
        self.features = np.hstack(
            [self.features, feature]
            if self.features is not None else feature)

    def _extract_mfcc(self, n_mfcc=12):
        """
        Extract MFCC mean and std_dev vectors for a clip.
        Appends (2*n_mfcc,) shaped vector to
        instance feature vector
        """
        mfcc = librosa.feature.mfcc(y=self.y,
                            sr=self.sr,
                            n_mfcc=n_mfcc)


        mfcc_mean = mfcc.mean(axis=1).T
        mfcc_std = mfcc.std(axis=1).T
        mfcc_feature = np.hstack([mfcc_mean, mfcc_std])
        self._concat_features(mfcc_feature)

    def _extract_spectral_contrast(self, n_bands=3):
        """
        Extract Spectral Contrast mean and std_dev vectors for a clip.
        Appends (2*(n_bands+1),) shaped vector to
        instance feature vector
        """
        spec_con = librosa.feature.spectral_contrast(y=self.y,
                                                     sr=self.sr,
                                                     n_bands=n_bands)

        spec_con_mean = spec_con.mean(axis=1).T
        spec_con_std = spec_con.std(axis=1).T
        spec_con_feature = np.hstack([spec_con_mean, spec_con_std])
        self._concat_features(spec_con_feature)


    def _extract_tempo(self):
        """
        Extract the BPM.
        Appends (1,) shaped vector to instance feature vector
        """
        tempo = librosa_tempo(y=self.y, sr=self.sr)
        self._concat_features(tempo)


    def extract_features(self, *feature_list, save_local=False):
        """
        Specify a list of features to extract, and a feature vector will be
        built for you for a given Audio sample.

        Option to save vector output locally as a .pkl file in data/ directory

        Currently supported features: 'mfcc', 'spectral_contrast', 'tempo'
        """
        for feature in feature_list:
            if feature == 'mfcc':
                self._extract_mfcc()
            elif feature == 'spectral_contrast':
                self._extract_spectral_contrast()
            elif feature == 'tempo':
                self._extract_tempo()
            else:
                raise KeyError('Feature type not understood; see docstring.')

        if save_local:
            self._save_local(mem_clean=True)


    def _save_local(self, mem_clean=True):
        self.local_path = self.path.split('/')[-1]
        self.local_path = self.local_path.replace('.mp3', '').replace(' ', '')

        with open(f'data/{self.local_path}.pkl', 'wb') as out_f:
            pickle.dump(self, out_f)

        if mem_clean:
            self.y = None

    #Phần mở rộng
    def _extract_chroma(self):
        chroma = librosa.feature.chroma_stft(y=self.y, sr=self.sr)
        chroma_mean = chroma.mean(axis=1).T
        chroma_std = chroma.std(axis=1).T
        chroma_feature = np.hstack([chroma_mean, chroma_std])
        self._concat_features(chroma_feature)

    def _extract_zero_crossing_rate(self):
        zcr = librosa.feature.zero_crossing_rate(y=self.y)
        zcr_mean = zcr.mean(axis=1).T
        zcr_std = zcr.std(axis=1).T
        zcr_feature = np.hstack([zcr_mean, zcr_std])
        self._concat_features(zcr_feature)

    def _extract_spectral_rolloff(self):
        rolloff = librosa.feature.spectral_rolloff(y=self.y, sr=self.sr)
        rolloff_mean = rolloff.mean(axis=1).T
        rolloff_std = rolloff.std(axis=1).T
        rolloff_feature = np.hstack([rolloff_mean, rolloff_std])
        self._concat_features(rolloff_feature)

    def extract_features(self, *feature_list, save_local=False):
        for feature in feature_list:
            if feature == 'mfcc':
                self._extract_mfcc()
            elif feature == 'spectral_contrast':
                self._extract_spectral_contrast()
            elif feature == 'tempo':
                self._extract_tempo()
            elif feature == 'chroma':
                self._extract_chroma()
            elif feature == 'zcr':
                self._extract_zero_crossing_rate()
            elif feature == 'rolloff':
                self._extract_spectral_rolloff()
            else:
                raise KeyError('Feature type not understood; see docstring.')

        if save_local:
            self._save_local(mem_clean=True)