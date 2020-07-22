#pragma once

#include <string>
#include <cstring>
#include <iostream>

namespace magmadnn {

class Arguments {
public:
   Arguments():
      enable_shortcut(true)
   {}   
   
      int parse(std::string const& context, int argc, char** argv) {

      std::string help = 
         "Usage: " + context + " [options]\n" 
         R"(
Options:
--disable-shorcut     Disable shorcut in residual layers        
)";


         
      for( int i = 1; i < argc; ++i ) {

         if ( !strcmp("--help", argv[i]) ) {

            std::cout << help;
            return 1;
         }

         // Resnet
         else if ( !strcmp("--disable-shorcut", argv[i])) {
            enable_shortcut = false;
            std::cout << "Resnet: disable shortcuts"  << std::endl;
         }

      }

   }

   // Enable shorcut for residual layers. If set to `false`, simply
   // implements plain convolutional networks.
   bool enable_shortcut;
};

} // End of namespace magmadnn
