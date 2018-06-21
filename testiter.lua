file = io.open("testdata.csv", "w")

for x1=0.4,1.5,0.01 do	
	for x2=0.4,1.5,0.01 do
		S = S4.NewSimulation()
		S:SetLattice({1,0}, {0,0})
		S:SetNumG(5)
		S:AddMaterial("Ramanium1", {-100, 0.5})
		S:AddMaterial("Ramanium2", {12.25,0.5})
		S:AddMaterial("Ramanium3", {4,0})
		S:AddMaterial("Ramanium4", {2,0})
		S:AddMaterial("Vacuum", {1,0})
		S:AddLayer('AirAbove', 0, 'Vacuum')
		S:AddLayer('Slab', x1, 'Ramanium4')
		S:AddLayer('Slab2', x2, 'Ramanium3')
		S:AddLayer('Slab3', 0.5, 'Ramanium2')
		S:AddLayer('Slab4', 0.5, 'Ramanium1')
		S:AddLayerCopy('AirBelow', 0, 'AirAbove')
		S:SetExcitationPlanewave({0,0}, {0,0}, {1,0})
		S:SetFrequency(0.5)
	        forward, backward = S:GetPoyntingFlux('AirAbove', 0)
                t = {backward/forward,x1,x2}
                print(table.concat(t,","))
                file:write(table.concat(t,","), "\n")
	end
end

file:close()
