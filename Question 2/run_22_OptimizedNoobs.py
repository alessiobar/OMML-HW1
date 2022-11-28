import time
import pickle
from functions_22 import *
import pandas as pd
Method = "Gradient Descent"

df = np.array([[-1.68156303e-02,  1.49361613e+00, -9.96793759e-01],[-9.54450459e-02,  2.74278733e+00, -1.16915490e+00],
               [-1.90014212e+00,  6.74755716e-01,  2.56156847e-01],[-9.82228845e-01,  1.46448304e+00,  8.80567438e-01],
               [ 9.19084459e-01,  1.49532170e+00, -8.63522815e-01],[-1.06740703e+00,  1.58266813e+00,  1.46630109e+00],
               [-1.98190522e+00, -7.27572190e-01, -1.64370083e+00],[ 6.07004659e-01,  1.75515471e+00, -9.16813026e-01],
               [ 1.27023399e+00,  5.93176713e-01, -8.46374447e-01],[-1.32424366e+00,  2.01166491e+00,  3.33286164e+00],
               [ 1.24797982e+00,  7.97725193e-02, -7.49105582e-01],[ 8.81936833e-01,  1.17980193e+00, -9.40206993e-01],
               [ 9.21811080e-01,  2.19912631e+00, -1.05706402e+00],[-1.96801001e+00,  1.86164370e+00,  4.29320588e+00],
               [-1.62786016e+00,  1.08877437e+00,  1.81597481e+00],[-1.53622509e+00,  7.61140748e-01,  4.14221719e-01],
               [-7.00634706e-01,  2.16646540e+00,  4.26592132e-01],[-4.05942800e-01, -1.76274110e+00, -3.75900543e-01],
               [-4.85903347e-01, -1.19877494e+00, -9.01269074e-01],[-1.01563690e-01, -2.53595702e+00, -1.52700539e-01],
               [-1.92522560e+00,  1.23841609e+00,  2.59430982e+00],[-4.67857624e-01,  6.73234125e-02, -2.89733772e+00],
               [ 1.18221774e-01, -1.19074109e+00, -7.26252644e-01],[-1.59436926e+00, -2.27469726e+00, -1.14146483e+00],
               [-1.25412129e+00,  1.17475491e+00,  1.14805912e+00],[ 1.90437420e+00, -1.81018790e+00,  3.30583096e+00],
               [-2.09456241e-01,  2.31986077e+00, -7.96093818e-01],[ 1.40727829e+00,  1.24927346e+00, -8.03548631e-01],
               [-1.06215886e+00, -1.52718060e+00, -7.79983507e-01],[-1.18198589e+00, -2.45566873e+00, -7.83566930e-01],
               [ 4.62883214e-01,  8.82908650e-01, -1.60371191e+00],[-2.98170061e-01, -2.91216566e+00, -2.73059645e-01],
               [-1.18075375e-01, -1.52004925e+00, -4.11639457e-01],[-8.69902888e-01,  4.58454498e-01, -1.62073573e+00],
               [-1.24305407e+00,  9.41413906e-01,  3.94782966e-01],[-4.87644386e-01,  1.01099007e+00, -1.27059599e+00],
               [ 1.75961197e+00,  1.22852403e-01, -3.47913922e-01],[ 1.94072882e+00,  2.42165816e+00, -1.46522185e+00],
               [ 2.30626541e-01,  1.08261849e+00, -1.45685498e+00],[-1.34280284e+00,  5.72017596e-01, -4.47406170e-01],
               [ 1.01562810e+00,  1.86383126e+00, -9.30341719e-01],[-1.50507580e+00, -2.73311749e+00, -1.08418270e+00],
               [-3.88161037e-02,  2.04738757e+00, -8.71027969e-01],[-3.17923780e-01,  1.57443989e+00, -7.19081865e-01],
               [ 5.89415726e-01,  1.17546911e+00, -1.12195901e+00],[-1.25280447e+00,  2.50811163e+00,  2.62461876e+00],
               [-5.15641356e-01, -2.84658597e+00, -3.71476247e-01],[-1.23358567e+00,  2.05984111e+00,  2.92082102e+00],
               [ 6.06014377e-01, -1.46935721e+00,  3.14306228e-01],[ 1.61774151e+00,  7.71090477e-01, -7.17285845e-01],
               [-1.80712693e+00,  2.36746829e+00,  4.24108577e+00],[ 5.19056923e-02, -2.39416425e-01, -3.08017274e+00],
               [ 1.78677892e+00, -5.57928089e-01,  1.12914212e+00],[ 1.65259668e+00,  1.34241847e+00, -8.68200075e-01],
               [-1.56295376e-01, -1.75194468e+00, -2.67896936e-01],[ 1.48359307e+00,  2.86681788e+00, -1.52292370e+00],
               [-1.58477600e+00,  1.34216769e+00,  2.72442061e+00],[-6.87857448e-01,  1.09876701e+00, -7.19759813e-01],
               [ 1.36091242e+00, -9.53476739e-01,  1.63701035e+00],[-1.96686595e-01,  8.38535918e-01, -1.95564773e+00],
               [-1.59193910e+00,  2.17307055e+00,  4.34149064e+00],[-9.77217219e-01,  2.82760288e-01, -1.64342162e+00],
               [ 1.26567141e-01,  1.21453361e-01, -3.23240832e+00],[-1.22248490e+00, -1.96482846e-01, -1.51138517e+00],
               [ 1.49843991e+00, -7.41857208e-01,  1.35286135e+00],[-5.22899996e-01,  2.97771324e+00, -9.07656546e-01],
               [ 1.60512176e+00,  4.96025348e-01, -6.40389069e-01],[ 1.92311300e+00, -2.06908365e+00,  2.57580749e+00],
               [-1.19486970e+00,  1.35811908e+00,  1.46884317e+00],[ 4.40348665e-01,  1.52123554e+00, -9.68325787e-01],
               [-1.87719510e+00,  3.24666746e-01, -7.89609687e-01],[ 1.02890996e+00, -1.51602343e+00,  1.42175698e+00],
               [-2.49212669e-01, -1.08400073e+00, -1.09747018e+00],[ 1.05457640e+00, -1.61356479e+00,  1.51262250e+00],
               [-1.86738923e+00,  8.80661784e-01,  1.09134340e+00],[ 1.77374312e+00, -2.27339268e+00,  1.85455437e+00],
               [ 8.81616412e-02,  2.70736520e+00, -1.23760730e+00],[-7.61274705e-01, -1.25653012e+00, -8.41814073e-01],
               [ 4.48688938e-02,  2.62498495e+00, -1.16852277e+00],[ 8.64679475e-03, -7.00536225e-01, -2.00174066e+00],
               [ 1.18795038e+00, -1.11334166e+00,  1.48147256e+00],[ 1.39953505e+00, -6.59540845e-01,  9.05201100e-01],
               [ 1.88848659e+00,  2.62636714e+00, -1.54951218e+00],[ 1.78529537e+00, -1.63108795e+00,  3.50529725e+00],
               [-1.44774746e+00, -1.81004264e+00, -1.01133268e+00],[-6.23928230e-01,  1.23954689e+00, -6.03172787e-01],
               [-1.15518389e+00, -3.91225151e-01, -1.51802703e+00],[-1.19494041e+00, -1.25321980e+00, -9.57344040e-01],
               [ 1.60348031e+00,  2.12862042e+00, -1.17055210e+00],[ 1.30875249e+00,  9.81586335e-01, -7.94154335e-01],
               [-5.10318286e-01, -1.98245817e+00, -3.62671894e-01],[-8.70410759e-01,  1.76648926e-01, -1.97356816e+00],
               [-9.47630358e-01,  4.63733194e-01, -1.44027560e+00],[ 1.06821313e+00, -1.03687405e+00,  9.51366145e-01],
               [-9.31853964e-01,  1.67082426e-01, -1.85124672e+00],[-1.55341884e+00, -2.77288650e+00, -1.13373829e+00],
               [ 1.34767635e+00, -1.57379170e+00,  2.46898168e+00],[ 1.06175103e-01, -1.33955954e+00, -4.90679704e-01],
               [ 1.74138524e+00, -1.80486202e+00,  3.18717060e+00],[ 1.77083787e+00, -2.77099968e+00,  4.88470850e-01],
               [-1.32462830e+00,  2.08095368e+00,  3.37087234e+00],[ 5.17519086e-01,  1.31025293e+00, -1.05392751e+00],
               [-1.46089438e+00, -2.76172255e+00, -1.04578106e+00],[-1.01634535e+00,  1.12853696e+00,  2.40731399e-01],
               [ 3.22897738e-01,  2.64657736e+00, -1.24987067e+00],[-1.31004380e-01,  5.95892311e-01, -2.55494907e+00],
               [-5.02824027e-01,  8.39897697e-01, -1.60476098e+00],[-2.09117407e-01, -2.39487820e+00, -1.91672293e-01],
               [-1.66833143e+00,  1.94888112e+00,  4.44820644e+00],[ 7.94271595e-01,  3.19738317e-01, -1.72214616e+00],
               [-5.03158643e-01, -2.35126002e+00, -3.33633111e-01],[ 4.37068743e-01,  1.67898988e+00, -9.35049078e-01],
               [ 2.13890468e-01,  5.80761938e-01, -2.47428905e+00],[ 6.72312909e-01, -1.22918182e+00,  1.82774593e-01],
               [ 6.65480227e-01,  9.72093434e-01, -1.26381264e+00],[-1.01722413e+00,  1.30756847e+00,  6.73268767e-01],
               [ 6.62740642e-01, -1.09084912e+00, -6.97126604e-02],[ 8.24769593e-01, -8.48758013e-01, -1.58920829e-01],
               [-1.89382610e+00,  2.25962310e-02, -1.31048091e+00],[ 1.89714687e+00, -1.01904436e+00,  2.77117800e+00],
               [ 1.66274006e+00, -1.27306970e+00,  3.14153855e+00],[-7.19478554e-01, -2.84188304e+00, -4.86968272e-01],
               [-1.35184817e-01, -7.37660313e-01, -1.93354051e+00],[-1.16252253e+00,  1.56227062e+00,  1.84241338e+00],
               [ 4.33011465e-02, -2.43315801e+00, -8.35349944e-02],[ 1.15599085e+00, -2.37429434e+00,  8.62797716e-01],
               [-2.60271765e-01,  2.92035190e-04, -3.21837228e+00],[ 7.15535424e-01,  8.55586701e-01, -1.33840927e+00],
               [-1.85562547e+00, -1.52842923e+00, -1.42450765e+00],[ 1.80711655e+00,  7.19001666e-01, -7.17749077e-01],
               [ 5.82955997e-01,  4.69995659e-01, -2.06097268e+00],[ 7.98567932e-01,  1.51694824e+00, -8.85010339e-01],
               [ 1.26832907e-02,  1.35985796e+00, -1.12025611e+00],[ 6.50326687e-01,  1.67389171e+00, -9.06595744e-01],
               [ 4.71161370e-01,  9.45278877e-01, -1.49674455e+00],[ 1.02914448e+00,  2.64266640e+00, -1.29430782e+00],
               [ 1.03073979e+00,  1.63288541e+00, -8.68497114e-01],[ 7.07596908e-01,  1.65288377e+00, -8.98112660e-01],
               [ 1.90964511e+00, -2.17791106e+00,  2.21019391e+00],[-3.14593817e-01, -2.63340434e+00, -2.52980155e-01],
               [-1.84695569e+00,  9.23694082e-02, -1.19408596e+00],[ 2.82965311e-01,  2.22541951e+00, -1.03622200e+00],
               [-1.54713712e+00, -1.71869325e+00, -1.10687669e+00],[-1.11951575e+00, -1.08918841e+00, -1.02400548e+00],
               [-4.11187724e-01, -1.69003895e-01, -2.90766480e+00],[-1.96539905e+00, -5.19945257e-01, -1.63188021e+00],
               [ 4.63589123e-01, -1.14736320e+00, -3.70894257e-01],[-1.26783615e+00,  4.22220685e-01, -9.00332499e-01],
               [-8.58875043e-01,  2.35855806e+00,  9.97388448e-01],[ 1.90780528e+00,  2.93029626e+00, -1.73210053e+00],
               [ 9.38107640e-01, -9.14436152e-01,  3.04073782e-01],[-1.32586087e+00,  8.87629983e-01,  4.37329613e-01],
               [ 1.59673003e+00, -8.61782347e-01,  1.91925904e+00],[ 1.72092874e+00, -2.10138607e+00,  2.36819687e+00],
               [ 1.67246670e+00, -2.65713298e+00,  7.26803348e-01],[-3.29922228e-02, -1.53635466e+00, -3.52320594e-01],
               [-1.41583611e+00, -5.87054280e-01, -1.33122625e+00],[-1.93139599e+00, -1.92532330e+00, -1.48929002e+00],
               [-1.05460131e+00,  9.89367074e-01,  1.81055740e-03],[-4.57226115e-01,  2.22807957e+00, -3.23860519e-01],
               [ 2.21375213e-01,  2.21908403e+00, -1.02324094e+00],[ 1.78635746e+00,  5.97875498e-01, -6.69408051e-01],
               [-5.79894247e-02, -2.06467548e+00, -1.30942758e-01],[-1.91112773e+00, -5.43639722e-01, -1.58342664e+00],
               [ 1.42454194e+00,  1.68085514e+00, -9.24162004e-01],[-1.62022960e+00,  2.80007265e-01, -8.47539794e-01],
               [-1.73059490e-01, -8.72856831e-01, -1.57800267e+00],[ 6.00221876e-01,  1.24484146e+00, -1.05953762e+00],
               [-8.17191762e-01,  1.62049235e+00,  5.16161729e-01],[ 2.30241767e-01, -2.90144650e+00, -8.06505420e-02],
               [-2.84723242e-01, -1.60106479e+00, -4.16099383e-01],[ 7.51999350e-01, -2.52082500e+00,  2.75946921e-01],
               [-4.14330760e-02,  1.21105794e+00, -1.29572993e+00],[-1.88768778e+00,  2.88502813e+00,  1.97857466e+00],
               [ 1.71231857e+00, -2.50938345e+00,  1.10894501e+00],[ 1.45263436e+00,  2.40274786e+00, -1.25788298e+00],
               [-1.16821714e+00,  1.68727286e+00,  2.12804263e+00],[ 1.84565311e+00, -1.21381681e-01,  1.93974490e-02],
               [ 4.01057929e-01, -1.98541675e+00,  1.72814447e-01],[ 4.09251107e-01,  1.26828015e+00, -1.13765195e+00],
               [-7.27993250e-01, -6.76031986e-01, -1.65672957e+00],[-9.93332218e-01,  1.71375967e+00,  1.36403026e+00],
               [ 1.82050294e-01, -8.07860616e-01, -1.56408144e+00],[ 1.44388280e+00, -2.62988753e+00,  6.85524791e-01],
               [ 1.41009576e+00,  2.11950474e+00, -1.10496718e+00],[ 1.38574283e+00,  2.07251297e+00, -1.07657327e+00],
               [-1.47246388e+00,  1.80294021e-01, -1.07656333e+00],[ 4.54195809e-02,  1.19544394e+00, -1.32732440e+00],
               [-1.42594800e+00,  8.81589193e-01,  6.41813635e-01],[-1.21636722e+00, -2.99047841e+00, -8.58110411e-01],
               [-5.77767051e-01, -2.69379626e+00, -3.90851143e-01],[ 1.85041914e+00, -6.52495919e-01,  1.47561716e+00],
               [-1.75982014e+00,  2.27741338e+00,  4.43595264e+00],[-1.33672023e+00, -9.49699525e-01, -1.15688736e+00],
               [-5.56308508e-01, -2.44549397e-01, -2.57424730e+00],[-3.19988858e-02, -2.43936465e+00, -1.16313072e-01],
               [ 1.16043414e+00,  9.93746382e-01, -8.52440974e-01],[-1.13826672e+00, -2.57654788e+00, -7.56070703e-01],
               [ 7.44410168e-01, -1.32120984e-01, -1.76912542e+00],[-1.98012765e-01,  1.98468123e+00, -7.28489212e-01],
               [-1.76047228e+00,  6.01472541e-01,  2.72195753e-02],[-1.44384721e+00, -2.75211422e+00, -1.02926626e+00],
               [ 9.32537607e-01,  1.02255731e+00, -9.84681557e-01],[-7.13423721e-01,  1.20268723e+00, -4.62246887e-01],
               [ 9.05667013e-01, -8.81367952e-02, -1.36512097e+00],[-1.65747914e+00,  6.22194997e-01,  6.06147778e-02],
               [ 1.21832240e+00,  1.33531957e-01, -8.32908055e-01],[-6.53085571e-01,  1.82872082e+00,  1.63624358e-01],
               [ 1.04984508e+00,  2.23302230e+00, -1.08635868e+00],[-1.17692505e+00,  6.48043868e-01, -5.71150282e-01],
               [ 1.12574833e+00, -2.17323966e+00,  1.14091981e+00],[ 1.65236750e+00, -7.79048383e-01,  1.73749614e+00],
               [-1.56520201e+00,  2.61776135e+00,  3.30838832e+00],[ 1.54151906e+00,  1.49452610e+00, -8.87874375e-01],
               [ 3.64904104e-01,  2.60800797e+00, -1.23221956e+00],[ 5.53842546e-01, -1.60558681e+00,  3.06485955e-01],
               [-1.40360625e-02, -9.91855809e-01, -1.24147933e+00],[-1.39732616e+00, -1.92145077e+00, -9.61055247e-01],
               [-1.25513259e+00, -1.00356166e+00, -1.10590235e+00],[-1.82942846e+00,  2.72895097e+00,  2.88544525e+00],
               [ 4.29155438e-01, -5.52812169e-01, -1.83935075e+00],[ 1.23418496e+00, -3.07361637e-01, -2.77746176e-01],
               [-4.06751593e-01, -1.50762673e-01, -2.93196849e+00],[-5.30525063e-01,  2.53925470e+00, -3.26289945e-01],
               [ 1.32471333e+00,  2.19967260e+00, -1.12119359e+00],[-1.60345952e+00,  9.37445357e-03, -1.22753634e+00],
               [ 1.94843638e+00, -2.82020137e+00,  3.43010067e-01],[-1.87624493e+00,  1.68312491e+00,  4.11362777e+00],
               [-1.02836827e+00,  8.15042376e-01, -5.08775433e-01],[-1.93308468e+00,  2.52138337e-01, -9.82398520e-01],
               [-6.25535291e-01, -5.20026010e-01, -2.04798133e+00],[-3.40277048e-01,  9.30903272e-01, -1.62938447e+00],
               [ 1.62456601e+00,  5.69654206e-01, -6.61993011e-01],[-1.02579309e+00,  9.72776024e-01, -1.22027698e-01],
               [-9.69877460e-01, -2.69538868e+00, -6.39061458e-01],[-1.26414715e+00, -2.99675774e+00, -8.98411006e-01],
               [ 1.63511979e+00, -1.33697048e+00,  3.17880959e+00],[ 2.31087994e-01, -2.15727589e+00,  3.21696039e-02],
               [-1.10657745e+00, -8.60804139e-01, -1.19536215e+00],[ 1.09207101e+00, -1.11792906e+00,  1.18390622e+00],
               [ 3.34575450e-01, -1.22321148e+00, -4.35718640e-01],[ 1.06613656e+00,  1.81454578e+00, -9.17534505e-01],
               [ 3.60099090e-01, -1.32856830e+00, -2.40624175e-01],[ 1.99710005e+00,  2.27903575e+00, -1.42072469e+00],
               [ 1.48624007e-01,  2.38640674e+00, -1.07716109e+00],[-9.68999071e-01,  2.35468251e+00,  1.50983809e+00],
               [-1.17729746e+00, -1.82910546e+00, -7.91381799e-01],[-4.98832722e-01, -1.14012699e+00, -9.88764840e-01],
               [-1.13478302e+00,  2.63514652e-01, -1.36910106e+00],[ 1.39592683e+00,  2.31252049e+00, -1.19532342e+00]])

X, y = df[:,[0,1]], df[:,2]
idx = np.random.choice(X.shape[0], int(0.75*X.shape[0])-1, replace=False)
mask = np.ones((X.shape[0]), dtype=bool)
mask[idx] = False #i.e., not idx
X_train, Y_train, X_test, Y_test = X[idx].T, y[idx].reshape((186, 1)), X[mask].T, y[mask].reshape((64, 1))
if __name__ == '__main__':
    start = time.time()
    params, train_errors, y, num_grad, num_eval = fit(X_train, Y_train, N = 30, sigma = 1.2, r = 0.01, learning_rate = 0.01, number_of_iterations = 2000, supervised = False)
    with open ('weights22.pkl', 'wb') as f:
        pickle.dump(params,f)
    comp_time = time.time() - start
    tr_err = error_test(y, Y_train)
    y_tst, F, _  = forwardPropagation(X = X_test, params = params)
    tst_err = error_test(y_tst, Y_test)
    res = { "Number of neurons N chosen": params["N"],  "Value of spread chosen": params["sigma"], "Value of ro chosen": params["r_c"]/X_train.shape[1], "Optimization solver chosen": Method, "Number of function evaluations": num_eval, "Number of gradient evaluations": num_grad, "Time for optimizing the network": comp_time, "Training Error": tr_err, "Test Error": tst_err}
    print(res)
