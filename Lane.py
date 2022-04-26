class Lane():
    def __init__(self):
        self.leftCurve = None 
        self.rightCurve = None  
        self.radiusOfCurvature = None 
        self.laneOffset = None 
     
    def getRightCurve(self):
        return self.rightCurve
    def getLeftCurve(self):
        return self.leftCurve    
    def getRadiusOfCurvature(self):
        return self.radiusOfCurvature  
    def getLaneOffset(self):
        return self.laneOffset
    
    def setRightCurve(self, value):
        self.rightCurve = value  
    def setLeftCurve(self, value):
        self.leftCurve = value    
    def setRadiusOfCurvature(self, value):
        self.radiusOfCurvature = value   
    def setLaneOffset(self, value):
        self.laneOffset = value