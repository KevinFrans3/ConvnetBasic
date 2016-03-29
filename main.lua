require "nn"
require "image"

net = nn.Sequential()
--
net:add(nn.SpatialConvolution(3,32,5,5))
net:add(nn.ReLU())
net:add(nn.SpatialMaxPooling(2,2,2,2))
net:add(nn.SpatialConvolution(32,64,5,5))
net:add(nn.ReLU())
net:add(nn.SpatialMaxPooling(2,2,2,2))
net:add(nn.View(64*5*5))
net:add(nn.Dropout(0.5))
net:add(nn.Linear(64*5*5,120))
net:add(nn.ReLU())
net:add(nn.Dropout(0.5))
net:add(nn.Linear(120,84))
net:add(nn.ReLU())
net:add(nn.Dropout(0.5))
net:add(nn.Linear(84,10))
net:add(nn.LogSoftMax())

net = torch.load("net.dat")

print ("Stucture: " .. net:__tostring() )

trainset = torch.load('cifar10-train.t7')
testset = torch.load('cifar10-test.t7')
classes = {'airplane', 'automobile', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck'}

print(#trainset.data[1])
print(#image.lena())
-- image.display(image.scale(trainset.data[99],320,320,"simple"))


setmetatable(trainset,
    {__index = function(t, i)
                    return {t.data[i], t.label[i]}
                end}
    );

trainset.data = trainset.data:double() -- convert the data from a ByteTensor to a DoubleTensor.

function trainset:size()
    return self.data:size(1)
end

mean = {}
stdv = {}
for i=1,3 do
    mean[i] = trainset.data[{{},{i},{},{}}]:mean()
    trainset.data[{{},{i},{},{}}]:add(-mean[i])
    stdv[i] = trainset.data[{{},{i},{},{}}]:std()
    trainset.data[{{},{i},{},{}}]:div(stdv[i])
end

testset.data = testset.data:double()   -- convert from Byte tensor to Double tensor
for i=1,3 do -- over each image channel
    testset.data[{ {}, {i}, {}, {}  }]:add(-mean[i]) -- mean subtraction
    testset.data[{ {}, {i}, {}, {}  }]:div(stdv[i]) -- std scaling
end

criterion = nn.ClassNLLCriterion()
trainer = nn.StochasticGradient(net, criterion)


function testit()
    local correct = 0
    for i=1,10000 do
        local groundtruth = testset.label[i]
        local prediction = net:forward(testset.data[i])
        local confidences, indices = torch.sort(prediction, true)  -- true means sort in descending order
        if groundtruth == indices[1] then
            correct = correct + 1
        end
    end
    print(correct, 100*correct/10000 .. ' %  on test')

    correct = 0
    for i=1,10000 do
        local groundtruth = trainset.label[i]
        local prediction = net:forward(trainset.data[i])
        local confidences, indices = torch.sort(prediction, true)  -- true means sort in descending order
        if groundtruth == indices[1] then
            correct = correct + 1
        end
    end
    print(correct, 100*correct/10000 .. ' %  on training')
end


testit()

function iter(self,iteration,currentError)
    print("on iter: " .. iteration .. " error: " .. currentError)
    -- testit()
end
trainer.hookIteration = iter
trainer.learningRate = 0.001
trainer.maxIteration = 15 -- just do 5 epochs of training.
trainer:train(trainset)

torch.save('net.dat', net)

print(classes[testset.label[100]])
-- image.display(image.scale(testset.data[100],320,320,"simple"))
predicted = net:forward(testset.data[100])
for i=1,predicted:size(1) do
    print(classes[i], predicted[i])
end

testit()
