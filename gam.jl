using PyPlot
using DataFrames

function main()

	df = readtable("motorcycle.csv")
	scatter(df[:times], df[:accel], color="0.75", alpha=0.8)
	xlabel("times")
	ylabel("Acceleration")

	k = 20 #length(df[:times])

	knots = quantile(df[:times], Array(linspace(0, 1, k)))

	t = Array(df[:times])
	x = hcat(ones(length(df[:times])), t, apply_R(t, knots))

	S = zeros(k+2, k+2)	
	S[3:end, 3:end] = R(knots, knots)

	B = zeros(k+2, k+2)
	B[3:end, 3:end] = real(sqrtm(S[3:end, 3:end]))

	y_bar = fit_model(x, Array(df[:accel]), B, 10., k)
	plot(t, y_bar, color="k")


end


function R(x::Float64, z::Float64)

	r = (((z - 0.5)^2 - 1 / 12) * 
	    ((x - 0.5)^2 - 1 / 12) / 4 - 
	    ((abs(x - z) - 0.5)^4 - 0.5 * 
	    (abs(x - z)  - 0.5)^2 + 7 / 240) / 24)
	
	return r

end


function apply_R(x::Array{Float64, 1}, z::Array{Float64, 1})


	R_mat = ones(length(x), length(z))
	for (i, xi) in enumerate(x)
		for (j, zj) in enumerate(z)
		
			R_mat[i, j] = R(xi, zj)	

		end	
	end

	return R_mat


end



function fit_model(X::Array{Float64, 2}, Y::Array{Float64, 1}, 
		B::Array{Float64, 2}, lambda::Float64, k::Int64)

	Y_ = vcat(Y, zeros(k+2))
	X_ = vcat(X, sqrt(lambda) * B)

	return (X * inv(X_' * X_) * X_') * Y_


end

