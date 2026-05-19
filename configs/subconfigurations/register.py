import configs.subconfigurations.controller.registration as controller
import configs.subconfigurations.cpg.registration as cpg
import configs.subconfigurations.genetic.registration as genetic
import configs.subconfigurations.logger.registration as logger
import configs.subconfigurations.random.registration as random
import configs.subconfigurations.simulation.registration as simulation

registered = False


def register():
    global registered
    if registered:
        return

    controller.register()
    cpg.register()
    genetic.register()
    logger.register()
    random.register()
    simulation.register()

    registered = True
