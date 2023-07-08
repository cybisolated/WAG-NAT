

class BaseDataset():

    @property
    def targets(self):
        return self.target

    @property
    def encoder_reals(self):
        return self.static_reals + self.time_varying_known_reals + self.time_varying_unknown_reals

    @property
    def encoder_cats(self):
        return self.static_categoricals + self.time_varying_known_categoricals + \
            self.time_varying_unknown_categoricals

    @property
    def decoder_reals(self):
        return self.static_reals + self.time_varying_known_reals

    @property
    def decoder_cats(self):
        return self.static_categoricals + self.time_varying_known_categoricals
