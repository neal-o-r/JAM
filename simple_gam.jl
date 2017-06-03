using PyPlot
using DataFrames

function main()

	df = readtable("motorcycle.csv")
	k = 20

	knots = quantile(df[:times], Array(linspace(0, 1, k)))

	t = Array(df[:times])
	x = hcat(ones(length(t)), t, R(t, knots))

	S = zeros(k+2, k+2)	
	S[3:end, 3:end] = R(knots, knots)

	B = zeros(k+2, k+2)
	B[3:end, 3:end] = real(sqrtm(S[3:end, 3:end]))

	y_bar, score, lambda = fit_model(x, Array(df[:accel]), B, k)
	println("Lambda: ", lambda, 
		"\nScore: ", score)

	plot_fit(df, y_bar)

end

function plot_fit(df::DataFrames.DataFrame, y_bar::Array{Float64, 1})

	figure()
	scatter(df[:times], df[:accel], color="0.75", alpha=0.8, s=3)
	xlabel("times")
	ylabel("Acceleration")

	t = Array(df[:times])
	plot(t, y_bar, color="k")

end

function R(x::Array{Float64, 1}, z::Array{Float64, 1})

	r = (((z' - 0.5).^2 - 1 / 12) .* 
	    ((x - 0.5).^2 - 1 / 12) / 4 - 
	    ((abs(broadcast(-, x, z')) - 0.5).^4 - 0.5 .* 
	    (abs(broadcast(-, x, z'))  - 0.5).^2 + 7 / 240) / 24)
	
	return r

end

function GCV_score(X::Array{Float64, 2}, Y::Array{Float64, 1}, 
			Y_hat::Array{Float64, 1})

	n = size(X)[2]
	In = (X * inv(X' * X) * X')
	res = sum((Y - Y_hat).^2) 
	H = In[1:n, 1:n]

	return n * res  / (n - trace(H))^2

end


function solve_ols(X::Array{Float64, 2}, Y::Array{Float64, 1}, 
	B::Array{Float64, 2}, 
	lambda::Float64, k::Int64)

	Y_ = vcat(Y, zeros(k+2))
	X_ = vcat(X, sqrt(lambda) * B)
	Y_bar = (X * inv(X_' * X_) * X_') * Y_

	g = GCV_score(X_, Y, Y_bar)
	
	return Y_bar, g

end

function fit_model(X::Array{Float64, 2}, Y::Array{Float64, 1}, 
		B::Array{Float64, 2}, k::Int64)

	lambdas = logspace(-3, 3, 20)
	
	gcvs = zeros(20)
	for (i, l) in enumerate(lambdas)
	
		_, s = solve_ols(X, Y, B, l, k)
		gcvs[i] = s

	end

	lambda = lambdas[indmin(gcvs)]

	figure()
	semilogx(lambdas, gcvs, "-o")
	ylabel("GCV Score")
	xlabel(L"$\lambda$")

	Y_hat, score = solve_ols(X, Y, B, lambda, k)
	
	return Y_hat, score, lambda

end

