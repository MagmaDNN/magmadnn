# get the testers
TESTING_FILES=$(cd bin && ls)

# run each tester
for i in $TESTING_FILES; do
    printf "==== TESTING %s ====\n" "$i"
    ./bin/$i
    if [ $? -ne 0 ]; then
        printf "\n%s TEST ERROR!\n" "$i"
    fi
    printf "\n"
done