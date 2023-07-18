import Lora.network_base
class NetworkModuleFull(Lora.network_base.NetworkModuleBase):
    @staticmethod
    def from_weights(weights: Lora.network_base.NetworkWeights):
        if all(x in weights.w for x in ["diff"]):
            return NetworkModuleFull(weights)

        return None

    def __init__(self, weights: Lora.network_base.NetworkWeights):
        super().__init__(weights)
        self.weight = weights.w.get("diff")

    def calc_updown(self, orig_weight):
        output_shape = self.weight.shape
        updown = self.weight.to(orig_weight.device, dtype=orig_weight.dtype)

        return self.finalize_updown(updown, orig_weight, output_shape)
