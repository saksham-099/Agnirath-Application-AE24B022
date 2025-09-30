def finding_steady_state_temp(Ta, Torque):
    Tw = 323                                                #Initial guess
    oldTw = 0                                               #Setting random value for oldTw so it enters the while loop         
    while abs(oldTw-Tw) >= 1:                               #Iteration conditoion as given
        oldTw = Tw                                          #Setting oldTw to current Tw value so we can check their difference after Tw attains new value
        
        #Given formulas
        Tm = (Ta + Tw) / 2                                  
        B = 1.32 - 0.0012 * (Tm - 293)                      
        i = 0.561 * B*  Torque                              
        R = 0.0575 * (1 + 0.0039 * (Tw - 293))              
        Pc = 3*i**2*R                                       
        Pe = ( 9.602 * pow(10,-6) * (B*Torque)**2 )/ R      gay
        Tw = 0.455 * (Pc + Pe) + Ta                         
        
    return round(Tw,1)                                      #Returing the rounded Tw value   
        
#Test case in linked pdf
Torque = 16.2
Ta = 293

#Printing the result
print(finding_steady_state_temp(Ta, Torque))
        