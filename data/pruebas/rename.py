import os

carpeta_principal = 'D:/RM_RENAL/CycleGAN/DATA/Motion/resp5s/pCASL_Breathing_51Meas_Noise4_0.2M0_Cortex-MedullaContrast5/SpinEcho_Model_143/M0/'

for carpeta in os.listdir(carpeta_principal): 
    ruta_carpeta = os.path.join(carpeta_principal, carpeta)

    if os.path.isdir(ruta_carpeta): 
        for archivo in os.listdir(ruta_carpeta): 
            ruta_archivo = os.path.join(ruta_carpeta, archivo)
            nuevo_nombre = os.path.join(ruta_carpeta, carpeta + '_' + archivo)
            os.rename(ruta_archivo, nuevo_nombre)