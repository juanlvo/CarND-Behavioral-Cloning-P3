from keras.models import Sequential, Model
import matplotlib.pyplot as plt

model = load_model('model.h5')

model.summary()
model.get_config()

capa_intemedia = Model(inputs = model.input, output = model.get_layer('dense_7', model.summary()).output)
ouput = capa_intermedia.predict(([image_array[None, :, :, :]])[0]
exit()