class NN(): 
    def __init__(self, dataset, optimizer, split: int = 70, epochs: int = 10, n_clicks: int = 0):
        print("Train Model")
        # Read the given CSV file, and view some sample records
        df = pd.read_csv(f"../datasets/c3o-experiments/{dataset}.csv")
        syndf = pd.read_csv("../dahboard/temp/sythetic_data.csv")
        df = df.drop(["machine_type" ], axis=1)
        syndf = syndf.drop(["machine_type" ], axis=1)
        dataset = df.values
        syndataset = syndf.values

        X = dataset[:,0:7]
        Y = dataset[:,-1]
        synX = syndataset[:,0:7]
        synY = syndataset[:,-1]

        x_scaler = preprocessing.StandardScaler()
        y_scaler = preprocessing.StandardScaler()
        synx_scaler = preprocessing.StandardScaler()
        syny_scaler = preprocessing.StandardScaler()
        
        print("finished standard Model")

        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=1-(0.01*split))


        X = x_scaler.fit_transform(X_train)
        Y= y_scaler.fit_transform(Y_train.reshape(-1, 1) )
        
        synX = synx_scaler.fit_transform(synX)
        synY= syny_scaler.fit_transform(synY.reshape(-1, 1) )
        
        model = Sequential([
            Dense(32, input_shape=(7,)),
            Dense(1)
        ])
        synmodel = Sequential([
            Dense(32, input_shape=(7,)),
            Dense(1)
        ])
        print("Compiled standard Model")

        model.compile(optimizer=optimizer,
              loss='mae',
              metrics=['accuracy'])
        synmodel.compile(optimizer=optimizer,
              loss='mae',
              metrics=['accuracy'])

        hist = model.fit(X, Y,
          batch_size=32, epochs=epochs)
        hist = synmodel.fit(synX, synY,
          batch_size=32, epochs=epochs)
        
        pred = model.predict(x_scaler.transform(X_test))
        synpred =synmodel.predict(synx_scaler.transform(X_test))

        scaled_y =y_scaler.inverse_transform(pred) 
        synscaled_y =y_scaler.inverse_transform(synpred) 
        
        def plot_scatter(x_values, y_values, y_values_2, stri, s1, s2):

            # Create a figure and axis
            fig, ax = plt.subplots()        
            x_values = list(range(1, len(y_values) + 1))

            # Plot the scatter plot
            ax.scatter(x_values, y_values, label='Predicted')
            ax.scatter(x_values, y_values_2, label='Actual')


            # Set labels for x and y axes
            ax.set_xlabel('Datasample')
            ax.set_ylabel('Runtime in ms')
            ax.legend()
            # Set a title for the plot
            if stri == "syn":
                ax.set_title(f'Neural Network trained with {s2} Synthetic Data')
            else: 
                ax.set_title(f'Neural Network trained with {s1} Private Data')

            # Show the plot
            fig.savefig(f'assets/{stri}{n_clicks}.png')
        
        plot_scatter(scaled_y, scaled_y, Y_test, "normal", X.shape, synX.shape)
        plot_scatter(synscaled_y, synscaled_y, Y_test, "syn",X.shape, synX.shape)
