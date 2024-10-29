from pointrix.controller.gs import DensificationController

class PoseFreeDensificationController(DensificationController):
    
    def f_step(self, **kwargs):
        return super().f_step(**kwargs)(self)