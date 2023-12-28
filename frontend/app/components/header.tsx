import Image from "next/image";

export default function Header() {
  return (
    <div className="z-10 max-w-5xl w-full items-center justify-between font-mono text-sm lg:flex">
      <p className="fixed left-0 top-0 flex w-full justify-center border-b border-gray-300 bg-gradient-to-b from-zinc-200 pb-6 pt-8 backdrop-blur-2xl dark:border-neutral-800 dark:bg-zinc-800/30 dark:from-inherit lg:static lg:w-auto  lg:rounded-xl lg:border lg:bg-gray-200 lg:p-4 lg:dark:bg-zinc-800/30">
       Grade 3&nbsp;
        
      </p>
      <div className="fixed bottom-0 left-0 flex h-48 w-full items-end justify-center bg-gradient-to-t from-white via-white dark:from-black dark:via-black lg:static lg:h-auto lg:w-auto lg:bg-none">
        <a
          href="https://birlabrainiacs.com/"
          className="flex items-center justify-center font-nunito text-lg font-bold gap-2"
        >
          <span>Birla Brainiacs</span>
          <Image
            className="rounded-xl"
            src="https://birlabrainiacs.com/wp-content/uploads/2023/06/BBL-NEW-LOGO-png-1-1-1024x258.png"
            alt="Birla Brainiacs Logo"
            width={256}
            height={256}
            priority
          />
        </a>
      </div>
    </div>
  );
}
