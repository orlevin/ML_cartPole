import training
import log_class
import run_model

log_class.init_log()

training_data = training.initial_data_for_traninig()
model = training.train_model(training_data)
model.save('test.model')

# if we want to load later the model
#model = dnn.bulidModel(4)
#model.load('test.model')

run_model.run_games(model)
