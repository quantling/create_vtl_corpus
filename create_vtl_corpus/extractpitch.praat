form Test command line calls
    sentence In_name
    sentence Out_name
endform

#writeInfoLine: "Extract pitch from: """, in_name$, """"

Read from file: in_name$
selectObject: 1
## auto, pitch floor Hz, pitch ceiling Hz
#To Pitch: 0, 100, 600
# time steps, pitch floor Hz
To Pitch (cc): 0, 100, 15, "no", 0.03, 0.45, 0.01, 0.35, 0.14, 600
selectObject: 2
Down to PitchTier
selectObject: 3
Save as PitchTier spreadsheet file: out_name$ 

#writeInfoLine: "Saved pitch tier to: """, out_name$, """"
