# get the testers
TESTING_FILES=$(cd bin && ls)

# run each tester
for i in $TESTING_FILES; do
    printf "==== TESTING %s ====\n" "$i"
    ./bin/$i
    if [ $? -ne 0 ]; then
        printf "\n%s %b \n" "$i" "\x1b[31m TEST ERROR! \x1b[0m"
    fi
    printf "\n"
done