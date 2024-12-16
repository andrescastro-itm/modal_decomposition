import pyMMF
import numpy as np
import matplotlib.pyplot as plt
import cv2


# from matplotlib import rc
# rc('figure', figsize=(18,9))
# rc('text', usetex=True)
# from IPython.display import display, Math
import cmath

class Generador:

    def electric_field(self,Wavelength):
        NA = 0.14
        radius = 5.5 # in microns
        areaSize = 2.7*radius # calculate the field on an area larger than the diameter of the fiber
        self.num_points = 2**8 # resolution of the window

        self.metodo1 = 'SA'


        B1 = 0.6961663
        B2 = 0.4079426
        B3 = 0.8974794
        C1 = (0.0684043)**2
        C2 = (0.1162414)**2
        C3 = (9.896161)**2
    

        # Longitud de onda al cuadrado
        wavelength_sq = Wavelength**2  

        # Cálculo del indice de refraccion
        n2 = np.sqrt(1 + (B1*wavelength_sq)/(wavelength_sq - C1) + (B2*wavelength_sq)/(wavelength_sq - C2) + (B3* wavelength_sq)/(wavelength_sq - C3))
        n1 = np.sqrt((NA**2)+(n2**2))


        # Create the fiber object
        profile = pyMMF.IndexProfile(npoints = self.num_points, areaSize = areaSize)

        # Initialize the index profile
        profile.initStepIndex(n1=n1,a=radius,NA=NA)

        # Instantiate the solver
        solver = pyMMF.propagationModeSolver()

        # Set the profile to the solver
        solver.setIndexProfile(profile)

        # Set the wavelength
        solver.setWL(Wavelength)

        # Estimate the number of modes for a graded index fiber
        Nmodes_estim = pyMMF.estimateNumModesSI(Wavelength,radius,NA,pola=1)

        print(f"Estimated number of modes using the V number = {Nmodes_estim}")

       
        modes_semianalytical = solver.solve(mode = 'SI', curvature = None)

        self.modes = {}
        idx = np.flip(np.argsort(modes_semianalytical.betas), axis=0)
        self.modes['SA'] = {'betas':np.array(modes_semianalytical.betas)[idx],'profiles':[modes_semianalytical.profiles[i] for i in idx]}
        
        #imode = 2
        #plt.figure()
        #plt.imshow(np.abs(self.modes[self.metodo1]['profiles'][imode].reshape([self.num_points]*2)),cmap='jet')
        #plt.gca().set_title("Ideal LP mode",fontsize=25)
        #plt.axis('off')
       # plt.show()

        

    def descom_modal(self,weight_LP01,phase_LP01,weight_LP11a,phase_LP11a,weight_LP11b,phase_LP11b):


        E_01 = self.modes[self.metodo1]['profiles'][0].reshape([self.num_points]*2)
        E_11a = self.modes[self.metodo1]['profiles'][1].reshape([self.num_points]*2)
        E_11b = self.modes[self.metodo1]['profiles'][2].reshape([self.num_points]*2)

        
        U_int = abs(weight_LP01 * cmath.exp(1j * phase_LP01) * E_01 + 
                    weight_LP11a * cmath.exp(1j * phase_LP11a) * E_11a + 
                    weight_LP11b * cmath.exp(1j * phase_LP11b) * E_11b)**2

        # Tamaño deseado para la imagen reescalada
        new_width = 127
        new_height = 127

        # Reescalar la imagen
        resized_im = cv2.resize(U_int, (new_width, new_height))

        # Encontrar el valor máximo de píxel en la imagen
        valor_maximo = np.max(resized_im)

        resized_image = resized_im/valor_maximo

        return resized_image