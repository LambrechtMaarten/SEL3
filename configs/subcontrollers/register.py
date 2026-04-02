import configs.subcontrollers.controller.registration as controller
import configs.subcontrollers.cpg.registration as cpg
import configs.subcontrollers.genetic.registration as genetic
import configs.subcontrollers.logger.registration as logger
import configs.subcontrollers.random.registration as random
import configs.subcontrollers.simulation.registration as simulation

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
